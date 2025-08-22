from multiprocessing import Process, Queue, Event
import time
import yaml

from fictrac_client import FicTracClient
from utils import csv_writer, noise_pattern_writer
from screeninfo import get_monitors  # for getting monitor info

from bowl_stimulate_class import *

if __name__ == "__main__":
    # Load and parse configuration
    with open("config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # Extract configuration with defaults
    duration = config["duration"]
    debug = config.get("debug", False)
    projector_width_pixels = config.get("projector_width_pixels", 1280)

    # Get noise settings with defaults
    noise = config.get("noise", {})
    noise_config = {
        "framerate": noise.get("framerate", 60),
        "pixel_size": noise.get("pixel_size", 5),
        "save_path": noise.get("save_path", "noise_patterns")
    }
    
    # Set up frame saving process
    noise_queue = Queue()
    save_path = noise_config["save_path"]
    save_process = Process(target=noise_pattern_writer, args=(noise_queue, save_path, debug))
    save_process.start()

    # Set up FicTrac if enabled
    use_fictrac = config.get("use_fictrac", False)
    
    if use_fictrac:
        fictrac_process = None
        fictrac_writer = None
        
        # Get FicTrac settings with defaults
        fictrac = config.get("fictrac", {})
        fictrac_config = {
            "host": fictrac.get("host", "127.0.0.1"),
            "port": fictrac.get("port", 3000),
            "output_csv": fictrac.get("output_csv", "fictrac.csv"),
            "debug": debug  # Use global debug
        }
        
        fictrac_queue = Queue()
        stop_event = Event()

        client = FicTracClient(
            host=fictrac_config["host"],
            port=fictrac_config["port"],
            queue=fictrac_queue,
            debug=fictrac_config["debug"],
            stop_event=stop_event
        )
        
        fictrac_process = Process(target=client.run)
        fictrac_writer = Process(target=csv_writer, args=(fictrac_queue, fictrac_config["output_csv"]))
        fictrac_process.start()
        fictrac_writer.start()

    # Get monitor information
    monitors = get_monitors()
    
    monitor_resolution = (monitors[0].width, monitors[0].height)
    projector_resolution = (monitors[1].width, monitors[1].height)

    print(f"[Main] Monitor setup detected: {len(monitors)} monitors")

    # Run the stimulus
    print(f"[Main] Running stimulus for {duration} seconds...")
    Arena = Stimulation_Pipeline(img_size=(360, 720,3),
                                fov_azi=(0,180), 
                                fov_ele=(0,140),
                                monitor_resolution=monitor_resolution,
                                projector_resolution=projector_resolution,
                                name = "Arena",
                                projector_width_pixels=projector_width_pixels,
                                debug=False)
    
    noise = ShowNoise(
        Arena, 
        pixelsize=noise_config["pixel_size"],
        framerate=noise_config["framerate"],
        save_queue=noise_queue,
        debug=debug
    )
    Arena.generate(noise.run, duration=duration, rot_offset=(0,0,0))
    
    # Clean up frame saving process
    print("[Main] Stimulus complete, stopping frame saving...")
    time.sleep(0.5)  # Small buffer to ensure last frames are in queue
    noise_queue.put("STOP")
    save_process.join(timeout=5)
    if save_process.is_alive():
        print("[Main] Force terminating frame saving process")
        save_process.terminate()
    
    # Clean up FicTrac if it was enabled
    if use_fictrac:
        print("[Main] Stopping FicTrac...")
        stop_event.set()
        fictrac_queue.put("STOP")
        
        for process in [fictrac_process, fictrac_writer]:
            process.join(timeout=5)
            if process.is_alive():
                print("[Main] Force terminating FicTrac process")
                process.terminate()

    # Clean up CV2 windows last
    try:
        cv2.destroyAllWindows()
        key = cv2.waitKey(1)
    except:
        pass

    print("[Main] All processes stopped.")

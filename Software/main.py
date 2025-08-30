from multiprocessing import Process, Queue, Event
import time
import yaml
from arduino import ArduinoCommunication
import json
import os
from datetime import datetime

from fictrac_client import FicTracClient
from utils import csv_writer, noise_pattern_writer
from screeninfo import get_monitors  # for getting monitor info

from bowl_stimulate_class import *

if __name__ == "__main__":

    # Unique run tag for this execution
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load and parse configuration
    with open("config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # Extract configuration with defaults
    duration = config["duration"]
    debug = config["debug"]
    projector_width_pixels = config["projector_width_pixels"]

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
    save_process = Process(target=noise_pattern_writer, args=(noise_queue, save_path, run_tag, debug))
    save_process.start()

    # Set up FicTrac if enabled
    use_fictrac = config["use_fictrac"]
    
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

    # Arena parameters
    img_size = (360, 720, 3)
    fov_azi = (0, 180)
    fov_ele = (0, 140)

    # Run the stimulus
    print(f"[Main] Running stimulus for {duration} seconds...")
    Arena = Stimulation_Pipeline(img_size=img_size,
                                fov_azi=fov_azi, 
                                fov_ele=fov_ele,
                                monitor_resolution=monitor_resolution,
                                projector_resolution=projector_resolution,
                                name = "Arena",
                                projector_width_pixels=projector_width_pixels,
                                debug=False)
    
    noise = ShowNoise(Arena,
                    pixelsize=noise_config["pixel_size"],
                    framerate=noise_config["framerate"],
                    debug=debug)
    
    # Hardware trigger to the 2P Microscope

    send_hardware_trigger = config["send_hardware_trigger"]
    
    if send_hardware_trigger:

        # Send hardware trigger to the 2p Microscope
        port = config["arduino_port"]
        baud_rate = config["arduino_baudrate"]

        arduino = ArduinoCommunication(port=port, baud_rate=baud_rate)
        hardware_trigger_2p = arduino.send_trigger()

    # Run and save from inside generate()
    Arena.generate(
        noise.run,
        duration=duration,
        rot_offset=(0, 0, 0),
        save_queue=noise_queue,  # <-- saving now happens inside generate()
    )

    # --- Dump experimental parameters to JSON manifest ---
    experiment_data = {
        "duration_s": duration,
        "debug": debug,
        "monitor_resolution": {
            "width": monitor_resolution[0],
            "height": monitor_resolution[1],
        },
        "projector_resolution": {
            "width": projector_resolution[0],
            "height": projector_resolution[1],
        },
        "projector_width_pixels": projector_width_pixels,
        "noise": {
            "framerate": noise_config["framerate"],
            "pixel_size": noise_config["pixel_size"],
            "save_path": noise_config["save_path"],
        },
        "use_fictrac": use_fictrac,
        "fictrac": (
            {
                "host": fictrac_config["host"],
                "port": fictrac_config["port"],
                "output_csv": fictrac_config["output_csv"],
                "debug": fictrac_config["debug"],
            } if use_fictrac else None
        ),
        "arena": {
            "img_size": img_size,
            "fov_azi": fov_azi,
            "fov_ele": fov_ele,
            "name": "Arena",
        },
        "stimulus": "ShowNoise",
        "arduino": {
            "port": port,
            "baud_rate": baud_rate,
            "hardware_trigger_2p_time": hardware_trigger_2p,
        } if send_hardware_trigger else None,
        # Helpful for matching JSON to saved frames on disk
        "outputs": {
            "noise_frames_dir": noise_config["save_path"],
            "fictrac_csv": (fictrac_config["output_csv"] if use_fictrac else None)
        }
    }

    # Where to save the manifest (next to noise frames by default)
    experiment_data_dir = noise_config.get("save_path", ".")
    os.makedirs(experiment_data_dir, exist_ok=True)

    # Timestamped, human-readable filename
    manifest_path = os.path.join(experiment_data_dir, f"experiment_data_{run_tag}.json")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2)

    # --- Teardown ---
    # Tell the writer we're done and wait for it to finish
    print("[Main] Stopping noise writer...")
    time.sleep(0.5)  # Small buffer to ensure last frames are in queue
    noise_queue.put("STOP")
    save_process.join()

    
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

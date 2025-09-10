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
    dark_screen_duration = config["dark_screen_duration"]

    # Get noise settings with defaults
    noise = config["noise"]
    noise_config = {
        "framerate": noise["framerate"],
        "pixel_size": noise["pixel_size"],
        "save_path": noise["save_path"],
        "file_name": noise["file_name"]
    }
    
    # Set up frame saving process
    noise_queue = Queue()
    save_path = noise_config["save_path"]
    file_name = noise_config["file_name"]
    save_process = Process(target=noise_pattern_writer, args=(noise_queue, save_path, file_name, run_tag, debug))
    save_process.start()

    # Warmup: put a dummy message
    noise_queue.put("WARMUP")
    
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

    # Hardware trigger to the 2P Microscope

    use_arduino = config["use_arduino"]
    
    if use_arduino:

        # Initialize Arduino communication
        port = config["arduino_port"]
        baud_rate = config["arduino_baudrate"]
        handshake_interval = config["arduino_handshake_interval"]

        arduino = ArduinoCommunication(port=port, baud_rate=baud_rate, handshake_interval=handshake_interval)
                

    # Run the stimulus
    print(f"[Main] Running stimulus for {duration} seconds...")
    Arena = Stimulation_Pipeline(img_size=img_size,
                                fov_azi=fov_azi, 
                                fov_ele=fov_ele,
                                monitor_resolution=monitor_resolution,
                                projector_resolution=projector_resolution,
                                name = "Arena",
                                projector_width_pixels=projector_width_pixels,
                                arduino=arduino,
                                dark_screen_duration=dark_screen_duration,
                                debug=False)
    
    noise = ShowNoise(Arena,
                    pixelsize=noise_config["pixel_size"],
                    framerate=noise_config["framerate"],
                    debug=debug)

    # --- Dump experimental parameters to JSON manifest BEFORE starting the experiment ---
    experiment_data = {
        "duration_s": duration,
        "debug": debug,
        "dark_screen_duration_s": dark_screen_duration,
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
            "file_name": noise_config["file_name"]
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
            "handshake_interval": handshake_interval,
            "hardware_trigger_2p_time": None,   # <-- Not yet triggered
        } if use_arduino else None,
        "outputs": {
            "fictrac_csv": (fictrac_config["output_csv"] if use_fictrac else None)
        }
    }

    # Save the manifest before the trigger
    experiment_data_dir = noise_config.get("save_path", ".")
    os.makedirs(experiment_data_dir, exist_ok=True)
    manifest_path = os.path.join(experiment_data_dir, f"{run_tag}_experiment_config.json")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2)

    # Send trigger to 2P microscope
    if use_arduino:
        hardware_trigger_2p = arduino.send_trigger()

    # Run generate()
    Arena.generate(
        noise.run,
        duration=duration,
        rot_offset=(0, 0, 0),
        save_queue=noise_queue,  # <-- frame aving happens inside generate()
    )

    # --- Update JSON with actual trigger time ---
    if use_arduino:
        experiment_data["arduino"]["hardware_trigger_2p_time"] = hardware_trigger_2p
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

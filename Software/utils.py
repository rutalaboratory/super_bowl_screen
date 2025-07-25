import csv

def csv_writer(queue, csv_path):
    print(f"[Writer] Starting CSV writer at {csv_path}")
    header = [
        "cnt", "dr_cam_x", "dr_cam_y", "dr_cam_z", "err",
        "dr_lab_x", "dr_lab_y", "dr_lab_z",
        "r_cam_x", "r_cam_y", "r_cam_z",
        "r_lab_x", "r_lab_y", "r_lab_z",
        "posx", "posy", "heading", "step_dir", "step_mag",
        "intx", "inty", "ts", "seq"
    ]
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while True:
            row = queue.get()
            if row == "STOP":
                break
            writer.writerow(row)
    print("[Writer] CSV writing complete and file closed.")

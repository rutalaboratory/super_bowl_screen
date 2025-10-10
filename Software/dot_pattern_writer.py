import h5py
import numpy as np

def dot_pattern_writer(queue, h5_path, debug=False):
    """
    Multiprocessing-compatible writer for dot stimulus data.
    Reads frame dicts from a queue and writes to an HDF5 file.
    Args:
        queue: multiprocessing.Queue with frame dicts or 'STOP' string.
        h5_path: Path to HDF5 file.
        debug: Print debug info if True.
    """
    import traceback
    with h5py.File(h5_path, 'a') as f:
        grp = f.require_group('frames')
        while True:
            try:
                item = queue.get()
                if item == "STOP":
                    if debug:
                        print("[dot_pattern_writer] Received STOP signal. Exiting.")
                    break
                if item == "WARMUP":
                    if debug:
                        print("[dot_pattern_writer] Received WARMUP signal.")
                    continue
                frame_data = item
                idx = str(frame_data['frame_number'])
                if idx in grp:
                    if debug:
                        print(f"[dot_pattern_writer] Frame {idx} already exists, skipping.")
                    continue
                frame_grp = grp.create_group(idx)
                # Save images
                frame_grp.create_dataset('original_image', data=frame_data['original_image'], compression='gzip')
                frame_grp.create_dataset('bowl_image', data=frame_data['bowl_image'], compression='gzip')
                # Save scalars
                frame_grp.attrs['timestamp'] = frame_data['timestamp']
                frame_grp.attrs['azimuthal_position'] = frame_data.get('azimuthal_position', None)
                frame_grp.attrs['elevation_position'] = frame_data.get('elevation_position', None)
                frame_grp.attrs['yaw'] = frame_data.get('yaw', None)
                frame_grp.attrs['total_elapsed_time'] = frame_data.get('total_elapsed_time', None)
                frame_grp.attrs['is_new_frame'] = frame_data.get('is_new_frame', None)
                # Save FicTrac data if present
                if frame_data.get('fictrac_data') is not None:
                    for k, v in frame_data['fictrac_data'].items():
                        try:
                            if np.isscalar(v):
                                frame_grp.attrs[f'fictrac_{k}'] = v
                            else:
                                frame_grp.create_dataset(f'fictrac_{k}', data=np.array(v))
                        except Exception as e:
                            if debug:
                                print(f"[dot_pattern_writer] Error saving FicTrac key {k}: {e}")
                if debug:
                    print(f"[dot_pattern_writer] Saved frame {idx}")
            except Exception as e:
                print(f"[dot_pattern_writer] Exception: {e}")
                traceback.print_exc()
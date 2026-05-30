#!/usr/bin/env python3
import os
import json
import numpy as np
import h5py
import multiprocessing as mp
from glob import glob
from tqdm import tqdm
import time
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            if 'joint_action/vector' not in f:
                return (file_path, False, None, None, None, None, "Missing joint_action/vector", [], [])
            
            action_data = f['joint_action']['vector'][:]
            
            try:
                l_pose = f['endpose']['left_endpose'][:]
                l_grip = f['endpose']['left_gripper'][:]
                r_pose = f['endpose']['right_endpose'][:]
                r_grip = f['endpose']['right_gripper'][:]
                
                if l_grip.ndim == 1: l_grip = l_grip.reshape(-1, 1)
                if r_grip.ndim == 1: r_grip = r_grip.reshape(-1, 1)
                
                state_data = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=1)
            except KeyError as e:
                return (file_path, False, None, None, None, None, f"Missing state key: {e}", [], [])

        act_min = np.min(action_data, axis=0)
        act_max = np.max(action_data, axis=0)
        
        PI = np.pi + 0.3
        joint_dims = [i for i in range(action_data.shape[1]) if i not in [6, 13]]
        act_outlier_dims = []
        for jd in joint_dims:
            if np.any(action_data[:, jd] > PI) or np.any(action_data[:, jd] < -PI):
                act_outlier_dims.append(jd)

        state_min = np.min(state_data, axis=0)
        state_max = np.max(state_data, axis=0)
        
        state_outlier_dims = []
        if np.any(np.abs(state_data) > 5):
            state_outlier_dims = np.where(np.any(np.abs(state_data) > 5, axis=0))[0].tolist()

        return (file_path, True, act_min, act_max, state_min, state_max, None, act_outlier_dims, state_outlier_dims)
                
    except Exception as e:
        return (file_path, False, None, None, None, None, str(e), [], [])

def process_task_files(task_info):
    task_name, task_dir, data_mode, delete_outliers = task_info
    
    hdf5_files = []
    
    if data_mode in ["clean", "both"]:
        pattern = os.path.join(task_dir, "demo_clean", "data", "*.hdf5")
        hdf5_files.extend(glob(pattern))
        
    if data_mode in ["randomized", "both"]:
        pattern = os.path.join(task_dir, "demo_randomized", "data", "*.hdf5")
        hdf5_files.extend(glob(pattern))

    results = {
        'task_name': task_name,
        'file_count': 0,
        'success_count': 0,
        'skipped_outlier_count': 0,
        'deleted_files': [],        
        'error_files': [],
        'outlier_files': [],
        'act_global_min': None,
        'act_global_max': None,
        'state_global_min': None,
        'state_global_max': None
    }

    for file_path in hdf5_files:
        (fpath, success, 
         act_min, act_max, state_min, state_max, 
         error_msg, act_outliers, state_outliers) = process_single_file(file_path)
        
        results['file_count'] += 1
        
        if success:
            if act_outliers:
                results['outlier_files'].append((fpath, "Action", act_outliers))
                results['skipped_outlier_count'] += 1
                
                if delete_outliers:
                    deleted = _delete_episode_files(fpath)
                    results['deleted_files'].extend(deleted)
                
                continue 
            
            results['success_count'] += 1
            
            if results['act_global_min'] is None:
                results['act_global_min'] = act_min
                results['act_global_max'] = act_max
            else:
                results['act_global_min'] = np.minimum(results['act_global_min'], act_min)
                results['act_global_max'] = np.maximum(results['act_global_max'], act_max)

            if results['state_global_min'] is None:
                results['state_global_min'] = state_min
                results['state_global_max'] = state_max
            else:
                results['state_global_min'] = np.minimum(results['state_global_min'], state_min)
                results['state_global_max'] = np.maximum(results['state_global_max'], state_max)
            
            if state_outliers:
                results['outlier_files'].append((fpath, "State", state_outliers))
        else:
            results['error_files'].append((fpath, error_msg))
            
    return results


def _delete_episode_files(hdf5_path):
    deleted = []
    
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)
        deleted.append(hdf5_path)
    
    # hdf5_path: .../demo_xxx/data/episode123.hdf5
    # json_path: .../demo_xxx/instructions/episode123.json
    data_dir = os.path.dirname(hdf5_path)           
    demo_dir = os.path.dirname(data_dir)             
    episode_name = os.path.splitext(os.path.basename(hdf5_path))[0]  
    
    json_path = os.path.join(demo_dir, "instructions", f"{episode_name}.json")
    if os.path.exists(json_path):
        os.remove(json_path)
        deleted.append(json_path)
    
    return deleted

def collect_stats_multiprocess(root_dirs, output_path, outlier_path, num_processes=16, data_mode="clean", delete_outliers=False):
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    print(f"Starting stats calculation for HDF5 dataset in: {root_dirs}")
    print(f"Data Mode: {data_mode}")
    if delete_outliers:
        print(f"\033[91m⚠ DELETE MODE: file delete!\033[0m")
    
    task_dirs = []
    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"Warning: Root directory {root_dir} does not exist. Skipping.")
            continue

        raw_dirs = os.listdir(root_dir)
        for d in raw_dirs:
            full_path = os.path.join(root_dir, d)
            if os.path.isdir(full_path):
                has_clean = os.path.exists(os.path.join(full_path, "demo_clean"))
                has_rand = os.path.exists(os.path.join(full_path, "demo_randomized"))
                if has_clean or has_rand:
                    task_dirs.append((d, full_path, data_mode, delete_outliers))
    
    print(f"Found {len(task_dirs)} task directories.")
    
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        task_results = list(tqdm(pool.imap(process_task_files, task_dirs), total=len(task_dirs), desc="Scanning Tasks"))

    total_files = 0
    total_success = 0
    all_errors = []
    all_outliers = []
    
    act_min = None; act_max = None
    state_min = None; state_max = None
    task_results.sort(key=lambda r: r['task_name'])
    per_task_stats = {}

    for res in task_results:
        task_name = res['task_name']
        total_files += res['file_count']
        total_success += res['success_count']
        all_errors.extend(res['error_files'])
        all_outliers.extend(res['outlier_files'])
        
        if res['act_global_min'] is not None:
            if act_min is None:
                act_min = res['act_global_min'].copy()
                act_max = res['act_global_max'].copy()
            else:
                act_min = np.minimum(act_min, res['act_global_min'])
                act_max = np.maximum(act_max, res['act_global_max'])

        if res['state_global_min'] is not None:
            if state_min is None:
                state_min = res['state_global_min'].copy()
                state_max = res['state_global_max'].copy()
            else:
                state_min = np.minimum(state_min, res['state_global_min'])
                state_max = np.maximum(state_max, res['state_global_max'])

        skipped = res.get('skipped_outlier_count', 0)
        if res['act_global_min'] is not None:
            per_task_stats[task_name] = {
                "file_count": res['file_count'],
                "valid_files": res['success_count'],
                "skipped_outlier_files": skipped,
                "action": {
                    "min": res['act_global_min'].tolist(),
                    "max": res['act_global_max'].tolist(),
                    "dim": len(res['act_global_min'])
                },
                "state": {
                    "min": res['state_global_min'].tolist() if res['state_global_min'] is not None else [],
                    "max": res['state_global_max'].tolist() if res['state_global_max'] is not None else [],
                    "dim": len(res['state_global_min']) if res['state_global_min'] is not None else 0
                }
            }
        elif skipped > 0:
            per_task_stats[task_name] = {
                "file_count": res['file_count'],
                "valid_files": 0,
                "skipped_outlier_files": skipped,
                "action": {"min": [], "max": [], "dim": 0},
                "state": {"min": [], "max": [], "dim": 0}
            }

    elapsed_time = time.time() - start_time

    total_skipped = sum(res.get('skipped_outlier_count', 0) for res in task_results)
    stat_dict = {
        "robotwin2": {
            "meta": {
                "data_mode": data_mode,
                "total_files": total_files,
                "valid_files": total_success,
                "skipped_outlier_files": total_skipped,
                "num_tasks": len(per_task_stats),
                "processing_time": elapsed_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "action": {
                "min": act_min.tolist() if act_min is not None else [],
                "max": act_max.tolist() if act_max is not None else [],
                "dim": len(act_min) if act_min is not None else 0
            },
            "state": {
                "min": state_min.tolist() if state_min is not None else [],
                "max": state_max.tolist() if state_max is not None else [],
                "dim": len(state_min) if state_min is not None else 0
            },
            "per_task": per_task_stats
        }
    }
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stat_dict, f, indent=4)
        print(f"\nStats saved to: {output_path}")
    except Exception as e:
        print(f"Error saving stats json: {e}")

    all_deleted = []
    for res in task_results:
        all_deleted.extend(res.get('deleted_files', []))

    try:
        with open(outlier_path, 'w', encoding='utf-8') as f:
            f.write(f"Outlier Report - Mode: {data_mode} - Total: {len(all_outliers)}\n")
            f.write("="*60 + "\n")
            f.write(f"Threshold: Action joint dims exceed ±π ({np.pi:.4f})\n")
            f.write(f"Delete mode: {'ON' if delete_outliers else 'OFF'}\n\n")
            for item in all_outliers:
                f.write(f"[{item[1]}] {item[0]} -> Dims: {item[2]}\n")
            if all_deleted:
                f.write(f"\n{'='*60}\n")
                f.write(f"DELETED FILES ({len(all_deleted)}):\n")
                for dp in all_deleted:
                    f.write(f"  {dp}\n")
        print(f"Outlier report saved to: {outlier_path}")
    except Exception as e:
        print(f"Error saving outlier report: {e}")

    print(f"\n{'='*30} SUMMARY {'='*30}")
    print(f"Mode: {data_mode}")
    print(f"Processed {total_success}/{total_files} files in {elapsed_time:.2f}s")
    print(f"Skipped {total_skipped} outlier files (action exceeds ±π)")
    if delete_outliers and all_deleted:
        print(f"\033[91m Delete {len(all_deleted)} (hdf5 + json)\033[0m")
    print(f"Tasks: {len(per_task_stats)}")

    if act_min is not None:
        print(f"\n--- Global Action (dim={len(act_min)}) ---")
        print(f"  Min: {np.round(act_min, 4)}")
        print(f"  Max: {np.round(act_max, 4)}")
    if state_min is not None:
        print(f"\n--- Global State (dim={len(state_min)}) ---")
        print(f"  Min: {np.round(state_min, 4)}")
        print(f"  Max: {np.round(state_max, 4)}")

    print(f"\n{'='*30} PER-TASK ACTION STATS {'='*30}")
    for task_name in sorted(per_task_stats.keys()):
        ts = per_task_stats[task_name]
        skipped = ts.get('skipped_outlier_files', 0)
        skip_str = f", \033[91mskipped {skipped} outlier\033[0m" if skipped > 0 else ""
        
        if ts['action']['dim'] == 0:
            print(f"\n  [{task_name}] ({ts['file_count']} files{skip_str}) — removed")
            continue
            
        a_min = np.array(ts['action']['min'])
        a_max = np.array(ts['action']['max'])
        a_range = a_max - a_min
        print(f"\n  [{task_name}] ({ts['valid_files']} valid / {ts['file_count']} total{skip_str})")
        print(f"    Action Min:   {np.round(a_min, 4)}")
        print(f"    Action Max:   {np.round(a_max, 4)}")
        print(f"    Action Range: {np.round(a_range, 4)}")

    print(f"\n{'='*30} PER-TASK STATE STATS {'='*30}")
    for task_name in sorted(per_task_stats.keys()):
        ts = per_task_stats[task_name]
        if ts['state']['dim'] > 0:
            s_min = np.array(ts['state']['min'])
            s_max = np.array(ts['state']['max'])
            print(f"\n  [{task_name}]")
            print(f"    State Min: {np.round(s_min, 4)}")
            print(f"    State Max: {np.round(s_max, 4)}")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Calculate HDF5 dataset statistics (State & Action)')
    parser.add_argument('--root_dir', type=str, nargs='+', 
                        default=["/home/yf/Desktop/Code/VLA/RoboTwin/RoboTwin/data-200"],
                        help='Root directory or directories containing task folders')
    
    parser.add_argument('--output_path', type=str, 
                        default="./stat-200-4.json",
                        help='Output JSON file path')
    
    parser.add_argument('--outlier_path', type=str, 
                        default="./outlier_files-200-4.txt",
                        help='Output text file for outliers')
    
    parser.add_argument('--num_processes', type=int, default=16,
                        help='Number of parallel processes')
    
    parser.add_argument('--data_mode', type=str, default='both', 
                        choices=['clean', 'randomized', 'both'],
                        help='Data split to process: clean, randomized, or both')
    
    parser.add_argument('--delete_outliers', action='store_true', default=True,
                        help='Delete action over ±π  episode (hdf5 + instruction json)')
    
    args = parser.parse_args()
    
    if args.delete_outliers:
        print("\n\033[91m⚠  --delete_outliers OPEN!")
        confirm = input("\n  Input 'yes': ")
        if confirm.strip().lower() != 'yes':
            print("  Cancel")
            return
        print()
    
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    
    collect_stats_multiprocess(
        args.root_dir, 
        args.output_path, 
        args.outlier_path, 
        args.num_processes,
        data_mode=args.data_mode,
        delete_outliers=args.delete_outliers
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
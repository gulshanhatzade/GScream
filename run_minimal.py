#!/usr/bin/env python3
"""
Minimal run script for GScream that optimizes storage usage.
This script minimizes disk usage by:
1. Disabling TensorBoard logging
2. Only saving the final model and outputs
3. Minimizing intermediate test renders
4. Limiting the number of checkpoints saved

Usage:
python run_minimal.py [--iterations ITERATIONS] [--dataset_path DATASET_PATH] [--output_path OUTPUT_PATH]
"""

import os
import sys
import argparse
import subprocess
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Run GScream with minimal storage usage')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of training iterations (default: 10000)')
    parser.add_argument('--dataset_path', type=str, 
                        default='/kaggle/input/spinnerf-dataset-processed/spinnerf_dataset_processed/1',
                        help='Path to the input dataset')
    parser.add_argument('--output_path', type=str, 
                        default='outputs/spinnerf_dataset/1/gscream/',
                        help='Path where to save outputs')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set environment variables to control storage usage
    os.environ['SAVE_TENSORBOARD'] = 'False'    # Disable TensorBoard (saves ~200MB per run)
    os.environ['SAVE_ALL_OUTPUTS'] = 'False'    # Don't save all rendering outputs
    os.environ['MINIMAL_TESTING'] = 'True'      # Only test at final iteration
    os.environ['MINIMAL_RENDERS'] = 'True'      # Skip intermediate renders
    os.environ['SAVE_ALL_ITERATIONS'] = 'False' # Only save final iteration outputs
    
    # Reference files needed for the run
    ref_files = ["./refs/1_out.png", "./refs/1_out_pred.npy"]
    
    # Check if reference files exist
    for ref_file in ref_files:
        if not os.path.exists(ref_file):
            print(f"Reference file {ref_file} not found. Please ensure it exists or download it.")
            sys.exit(1)
    
    # Create the minimal set of flags needed for training
    flags = [
        "--source_path", args.dataset_path,
        "--model_path", args.output_path,
        "--iterations", str(args.iterations),
        "--specified_ply_path", os.path.join(args.dataset_path, "sparse/0/points3D.ply"),
        "--load_mask", "--load_depth", "--load_norm",
        "--is_spin",
        "--ref_image_path", "./refs/1_out.png",
        "--ref_depth_path", "./refs/1_out_pred.npy",
    ]
    
    # Run the actual training command
    print(f"Running training with {args.iterations} iterations")
    print(f"Input dataset: {args.dataset_path}")
    print(f"Output directory: {args.output_path}")
    
    cmd = [sys.executable, "GScream-main/train.py"] + flags
    subprocess.run(cmd)
    
    # After training, clean up unnecessary files to further save space
    cleanup_unnecessary_files(args.output_path)
    
def cleanup_unnecessary_files(output_dir):
    """Clean up unnecessary files after training to save more space"""
    print("Cleaning up unnecessary files to save space...")
    
    # Paths to check for cleanup
    cleanup_paths = [
        os.path.join(output_dir, "train"),
        os.path.join(output_dir, "spiral")
    ]
    
    # Check and remove unnecessary directories
    for path in cleanup_paths:
        if os.path.exists(path):
            print(f"Removing {path}...")
            shutil.rmtree(path)
    
    # Remove all checkpoints except the final one
    checkpoint_dir = output_dir
    checkpoints = [f for f in os.listdir(checkpoint_dir) 
                   if f.startswith("chkpnt") and f.endswith(".pth")]
    
    if len(checkpoints) > 1:
        # Sort checkpoints by iteration number
        checkpoints.sort(key=lambda x: int(x.replace("chkpnt", "").replace(".pth", "")))
        
        # Keep only the latest checkpoint
        latest_checkpoint = checkpoints[-1]
        
        for checkpoint in checkpoints:
            if checkpoint != latest_checkpoint:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                print(f"Removing intermediate checkpoint: {checkpoint_path}")
                os.remove(checkpoint_path)

if __name__ == "__main__":
    main() 
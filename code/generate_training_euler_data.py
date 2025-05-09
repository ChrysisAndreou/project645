import read_bvh
import numpy as np
from os import listdir
import os
import read_bvh_hierarchy # It's good practice to import what's used, even if indirectly via read_bvh for skeleton


def generate_euler_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)
        print(f"Created directory: {tar_traindata_folder}")
    bvh_files = [f for f in listdir(src_bvh_folder) if f.endswith(".bvh")]
    print(f"Found {len(bvh_files)} BVH files in {src_bvh_folder}")
    for bvh_file_name in bvh_files:
        full_bvh_path = os.path.join(src_bvh_folder, bvh_file_name)
        print(f"Encoding Euler data from: {full_bvh_path}")
        # parse_frames returns the motion data directly (hip translation + all joint rotations)
        # This is suitable for the Euler representation.
        motion_data = read_bvh.parse_frames(full_bvh_path)
        
        if motion_data is not None and motion_data.shape[0] > 0:
            target_npy_path = os.path.join(tar_traindata_folder, bvh_file_name + ".npy")
            np.save(target_npy_path, motion_data)
            print(f"Saved Euler encoded data to: {target_npy_path} (shape: {motion_data.shape})")
        else:
            print(f"Skipping {bvh_file_name} due to empty or invalid motion data from parse_frames.")


def generate_bvh_from_euler_traindata(src_train_folder, tar_bvh_folder, standard_bvh_for_header):
    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)
        print(f"Created directory: {tar_bvh_folder}")
    npy_files = [f for f in listdir(src_train_folder) if f.endswith(".npy")]
    print(f"Found {len(npy_files)} .npy files in {src_train_folder}")
    for npy_file_name in npy_files:
        full_npy_path = os.path.join(src_train_folder, npy_file_name)
        print(f"Decoding Euler data from: {full_npy_path}")
        motion_data = np.load(full_npy_path)
        
        if motion_data is not None and motion_data.shape[0] > 0:
            # The npy_file_name is like "01.bvh.npy". We want "01.bvh" for the output.
            base_name = npy_file_name[:-4] # Remove .npy suffix
            target_bvh_path = os.path.join(tar_bvh_folder, base_name)
            
            read_bvh.write_frames(standard_bvh_for_header, target_bvh_path, motion_data)
            print(f"Saved reconstructed BVH to: {target_bvh_path}")
        else:
            print(f"Skipping {npy_file_name} due to empty or invalid loaded motion data.")

# Assuming scripts are run from the root of 'project645' directory
standard_bvh_file = "train_data_bvh/standard.bvh"
# weight_translation = 0.01 # This was in original template, seems unused for Euler direct processing
# skeleton, non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file) # Not strictly needed for these functions
# print('skeleton: ', skeleton) # Verbose

# As per instructions, use "salsa" data
bvh_dir_path = "train_data_bvh/salsa/"
euler_enc_dir_path = "train_data_euler/salsa/"  # Store encoded Euler data here
bvh_reconstructed_dir_path = "reconstructed_bvh_data_euler/salsa/" # Store reconstructed BVHs here

# Check if standard_bvh_file exists
if not os.path.exists(standard_bvh_file):
    print(f"ERROR: Standard BVH file for header ({standard_bvh_file}) not found. Please check the path.")
    # exit() # Or handle error appropriately

print(f"Starting Euler data generation for salsa dataset...")
print(f"Source BVH folder: {bvh_dir_path}")
print(f"Target Euler encoded data folder: {euler_enc_dir_path}")
print(f"Target reconstructed BVH folder (from Euler): {bvh_reconstructed_dir_path}")
print(f"Using standard BVH for header: {standard_bvh_file}")

# Encode data from bvh to Euler representation
if not os.path.exists(bvh_dir_path):
    print(f"ERROR: Source BVH directory {bvh_dir_path} not found. Please check the path.")
else:
    if os.path.exists(standard_bvh_file):
        generate_euler_traindata_from_bvh(bvh_dir_path, euler_enc_dir_path)
    else:
        print(f"Skipping encoding because standard BVH file for header is missing.")


# Decode from Euler representation to bvh for verification
if not os.path.exists(euler_enc_dir_path):
    print(f"INFO: Euler encoded data directory {euler_enc_dir_path} not found (perhaps encoding failed or was skipped). Skipping decoding.")
else:
    if os.path.exists(standard_bvh_file):
        generate_bvh_from_euler_traindata(euler_enc_dir_path, bvh_reconstructed_dir_path, standard_bvh_file)
    else:
        print(f"Skipping decoding because standard BVH file for header is missing.")

print("Euler data processing script finished.")

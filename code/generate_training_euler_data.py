import read_bvh
import numpy as np
from os import listdir
import os


def generate_euler_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    # TODO:

def generate_bvh_from_euler_traindata(src_train_folder, tar_bvh_folder):
    # TODO:


standard_bvh_file = "train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

print('skeleton: ', skeleton)

bvh_dir_path = None
euler_enc_dir_path = None
bvh_reconstructed_dir_path = None

# Encode data from bvh to positional encoding
generate_euler_traindata_from_bvh(bvh_dir_path, euler_enc_dir_path)

# Decode from positional to bvh
generate_bvh_from_euler_traindata(euler_enc_dir_path, bvh_reconstructed_dir_path)

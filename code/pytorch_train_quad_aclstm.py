import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse
import torch.nn.functional as F # For normalization
import read_bvh_hierarchy # Added import
from code import rotation_conversions as rc # Added import assuming it's in code/

# For raw BVH-derived quaternion data: Hip X pos is 0, Y pos is 1, Z pos is 2.
# Quaternions for rotations start from index 3.

Seq_len=100 # Default, can be overridden by args
Hidden_size = 1024 # Default, can be overridden by args
Condition_num=5
Groundtruth_num=5

class acLSTM(nn.Module):
    def __init__(self, in_frame_size=171, hidden_size=1024, out_frame_size=171): # Default in_frame_size for positional
        super(acLSTM, self).__init__()
        
        self.in_frame_size=in_frame_size
        self.hidden_size=hidden_size
        self.out_frame_size=out_frame_size
        
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
    
    def init_hidden(self, batch):
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h2= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        return  ([h0,h1,h2], [c0,c1,c2])
    
    def forward_lstm(self, in_frame, vec_h, vec_c):
        vec_h0,vec_c0=self.lstm1(in_frame, (vec_h[0],vec_c[0]))
        vec_h1,vec_c1=self.lstm2(vec_h[0], (vec_h[1],vec_c[1]))
        vec_h2,vec_c2=self.lstm3(vec_h[1], (vec_h[2],vec_c[2]))
        out_frame = self.decoder(vec_h2)
        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]
        return (out_frame,  vec_h_new, vec_c_new)
        
    def get_condition_lst(self,condition_num, groundtruth_num, seq_len ):
        gt_lst=np.ones((100,groundtruth_num))
        con_lst=np.zeros((100,condition_num))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]
        
    def forward(self, real_seq, condition_num=5, groundtruth_num=5):
        batch=real_seq.size()[0]
        seq_len=real_seq.size()[1]
        condition_lst=self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        (vec_h, vec_c) = self.init_hidden(batch)
        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda())
        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda())
        for i in range(seq_len):
            if(condition_lst[i]==1):
                in_frame=real_seq[:,i]
            else:
                in_frame=out_frame
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame),1)
        return out_seq[:, 1: out_seq.size()[1]]
    
    def calculate_loss(self, out_seq, groundtruth_seq):
        batch_size = out_seq.shape[0]
        num_elements = out_seq.shape[1]
        seq_len = num_elements // self.out_frame_size

        out_seq_reshaped = out_seq.reshape(batch_size, seq_len, self.out_frame_size)
        groundtruth_seq_reshaped = groundtruth_seq.reshape(batch_size, seq_len, self.out_frame_size)

        hip_pos_pred = out_seq_reshaped[..., :3]
        hip_pos_gt = groundtruth_seq_reshaped[..., :3]
        
        # Quaternions start from index 3. Each quaternion is 4 values.
        # (batch_size, seq_len, num_quads * 4)
        quad_pred_flat = out_seq_reshaped[..., 3:] 
        quad_gt_flat = groundtruth_seq_reshaped[..., 3:]

        loss_fn_mse = nn.MSELoss()
        loss_hip = loss_fn_mse(hip_pos_pred, hip_pos_gt)

        # Quaternion Loss: QL = 2 * arccos(|pred_q_normalized . gt_q|)
        # Reshape to (batch_size, seq_len, num_quads, 4) to process each quaternion
        num_quad_channels = quad_pred_flat.shape[-1]
        num_quads = num_quad_channels // 4

        quad_pred = quad_pred_flat.reshape(batch_size, seq_len, num_quads, 4)
        quad_gt = quad_gt_flat.reshape(batch_size, seq_len, num_quads, 4)

        # Normalize predicted quaternions
        quad_pred_normalized = F.normalize(quad_pred, p=2, dim=-1)
        # Ground truth quaternions are assumed to be normalized from preprocessing
        
        # Dot product: sum over the 4 components of the quaternion
        # (batch_size, seq_len, num_quads)
        dot_product = torch.sum(quad_pred_normalized * quad_gt, dim=-1)
        
        # Absolute value of the dot product
        dot_product_abs = torch.abs(dot_product)
        
        # Clamp to avoid acos errors with values slightly outside [-1, 1] due to precision
        epsilon = 1e-7
        dot_product_clamped = torch.clamp(dot_product_abs, -1.0 + epsilon, 1.0 - epsilon)
        
        angle_error = 2.0 * torch.acos(dot_product_clamped)
        loss_quad_ql = torch.mean(angle_error) # Mean over batch, seq_len, and num_quads
        
        total_loss = loss_hip + loss_quad_ql
        return total_loss

# Helper function to get Euler convention string for a joint
def get_euler_convention_for_joint(bone_name, skeleton_info):
    """Helper to extract Euler convention string (e.g., 'XYZ', 'ZYX') for a joint."""
    if bone_name not in skeleton_info:
        # This might happen if ordered_bones_with_rotations includes 'hip' but skeleton_info doesn't detail its rotation channels separately
        # Or if a bone name is somehow incorrect. For 'hip', rotations are part of its main channels.
        # Fallback or error handling might be needed if skeleton structure is very unusual.
        # For typical BVH from this project, hip rotation channels are with its position channels.
        # Let's assume standard processing where hip rotation channels are found like others if they exist.
        if bone_name == 'hip' and 'channels' in skeleton_info.get(bone_name, {}):
             pass # Proceed to get channels for hip
        else:
            print(f"Warning: Bone '{bone_name}' not found in skeleton_info for convention lookup.")
            return "ZYX" # Default or raise error

    joint_spec = skeleton_info[bone_name]
    joint_channels = joint_spec.get('channels', [])
    rotation_channels_for_joint = [ch for ch in joint_channels if 'rotation' in ch.lower()]
    
    convention_str = ""
    # The order of rotation channels in BVH defines the convention
    # e.g., Zrotation, Yrotation, Xrotation means an intrinsic ZYX rotation.
    for channel_name in rotation_channels_for_joint:
        convention_str += channel_name[0].upper() # Z, Y, X
    
    if not convention_str or len(convention_str) != 3:
        # print(f"Warning: Could not determine 3-char Euler convention for joint '{bone_name}'. Found: '{convention_str}'. Defaulting to ZYX.")
        return "ZYX" # A common default, but correctness depends on BVH source.
                       # This case should ideally not be hit if skeleton is parsed correctly from standard.bvh
                       # and all rotating joints have 3 rotation channels.
    return convention_str

def train_one_iteraton(real_seq_np, model, optimizer, iteration, save_dance_folder, print_loss=False, save_bvh_motion=True, standard_bvh_file=None, skeleton_info=None, ordered_bones_list=None): # Added new args
    dif_hip_x = real_seq_np[:, 1:, 0] - real_seq_np[:, :-1, 0]
    dif_hip_z = real_seq_np[:, 1:, 2] - real_seq_np[:, :-1, 2]
    real_seq_dif_hip_x_z_np = real_seq_np[:, :-1].copy()
    real_seq_dif_hip_x_z_np[:, :, 0] = dif_hip_x
    real_seq_dif_hip_x_z_np[:, :, 2] = dif_hip_z
    real_seq_cuda = torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np.tolist()).cuda())
    in_lstm_seq_len = real_seq_cuda.size()[1] - 1
    in_real_seq = real_seq_cuda[:, :in_lstm_seq_len]
    predict_groundtruth_seq_targets = real_seq_cuda[:, 1:in_lstm_seq_len+1]
    predict_groundtruth_seq_flat = predict_groundtruth_seq_targets.reshape(real_seq_np.shape[0], -1)
    predict_seq = model.forward(in_real_seq, Condition_num, Groundtruth_num)
    optimizer.zero_grad()
    loss = model.calculate_loss(predict_seq, predict_groundtruth_seq_flat)
    loss.backward()
    optimizer.step()

    if print_loss:
        print ("###########"+"iter %07d"%iteration +"######################")
        print ("loss: "+str(loss.detach().cpu().numpy()))

    if save_bvh_motion:
        current_frame_size = model.out_frame_size
        
        # --- Function to convert a sequence from quad to euler for BVH ---
        def convert_quad_seq_to_euler_bvh(quad_sequence_np, skeleton, ordered_bones, rc_module):
            num_frames = quad_sequence_np.shape[0]
            # Calculate expected Euler frame size: 3 (hip) + num_bones * 3 (euler angles)
            # This assumes all bones in ordered_bones_list contribute 3 Euler angles.
            euler_frame_size = 3 + len(ordered_bones) * 3 
            euler_sequence_list = []

            for frame_idx in range(num_frames):
                quad_frame = quad_sequence_np[frame_idx]
                euler_frame_data = []

                # Hip translation (first 3 values are already absolute due to reconstruction)
                euler_frame_data.extend(quad_frame[:3])

                current_quad_offset = 3 # Start after hip translation for quaternions

                for bone_name in ordered_bones:
                    if current_quad_offset + 4 > len(quad_frame):
                        print(f"Warning: Frame {frame_idx}, bone {bone_name}: Not enough data for quaternion. Skipping bone.")
                        # Add placeholder zeros for this bone's Euler angles to maintain structure
                        euler_frame_data.extend([0.0, 0.0, 0.0])
                        continue

                    quaternion_values = torch.tensor(quad_frame[current_quad_offset : current_quad_offset + 4], dtype=torch.float32)
                    
                    # Get Euler convention for this joint
                    convention = get_euler_convention_for_joint(bone_name, skeleton)

                    # Convert quaternion to Euler angles
                    rot_matrix = rc_module.quaternion_to_matrix(quaternion_values.unsqueeze(0)) # Add batch dim
                    euler_angles_rad = rc_module.matrix_to_euler_angles(rot_matrix, convention).squeeze(0) # Remove batch dim
                    euler_angles_deg = (euler_angles_rad * (180.0 / torch.pi)).tolist() # Use torch.pi
                    
                    euler_frame_data.extend(euler_angles_deg)
                    current_quad_offset += 4
                
                if len(euler_frame_data) != euler_frame_size:
                     print(f"Warning: Frame {frame_idx}: Mismatch in expected Euler frame size. Expected {euler_frame_size}, got {len(euler_frame_data)}. Padding/truncating.")
                     # Simple padding/truncating strategy
                     if len(euler_frame_data) < euler_frame_size:
                         euler_frame_data.extend([0.0] * (euler_frame_size - len(euler_frame_data)))
                     else:
                         euler_frame_data = euler_frame_data[:euler_frame_size]

                euler_sequence_list.append(euler_frame_data)
            
            return np.array(euler_sequence_list, dtype=np.float32)
        # --- End of conversion function ---

        # Ground truth sequence processing
        gt_seq_quad_np = np.array(predict_groundtruth_seq_targets[0].data.cpu().numpy()) # Shape: [lstm_seq_len, quad_frame_size]
        initial_hip_x_gt = real_seq_np[0, 0, 0] 
        initial_hip_z_gt = real_seq_np[0, 0, 2]
        
        gt_seq_abs_hip_quad_np = gt_seq_quad_np.copy()
        last_x_gt = initial_hip_x_gt
        last_z_gt = initial_hip_z_gt
        for frame_idx in range(gt_seq_abs_hip_quad_np.shape[0]):
            current_delta_x = gt_seq_abs_hip_quad_np[frame_idx, 0]
            current_delta_z = gt_seq_abs_hip_quad_np[frame_idx, 2]
            gt_seq_abs_hip_quad_np[frame_idx, 0] = last_x_gt + current_delta_x
            gt_seq_abs_hip_quad_np[frame_idx, 2] = last_z_gt + current_delta_z
            last_x_gt = gt_seq_abs_hip_quad_np[frame_idx, 0]
            last_z_gt = gt_seq_abs_hip_quad_np[frame_idx, 2]
        
        gt_seq_euler_bvh = convert_quad_seq_to_euler_bvh(gt_seq_abs_hip_quad_np, skeleton_info, ordered_bones_list, rc)
        if gt_seq_euler_bvh.size > 0: # Check if conversion produced data
            read_bvh.write_traindata_to_bvh(standard_bvh_file, os.path.join(save_dance_folder, "%07d"%iteration+"_gt.bvh"), gt_seq_euler_bvh)
        else:
            print(f"Warning: GT Euler sequence for iteration {iteration} is empty after conversion. Skipping save.")

        # Predicted sequence processing
        out_seq_quad_flat_np = np.array(predict_seq[0].data.cpu().numpy()) # Shape: [lstm_seq_len * quad_frame_size]
        out_seq_quad_np = out_seq_quad_flat_np.reshape(-1, current_frame_size) # Shape: [lstm_seq_len, quad_frame_size]
        
        initial_hip_x_pred = real_seq_np[0, 0, 0] # Predictions also start from same initial state context
        initial_hip_z_pred = real_seq_np[0, 0, 2]

        out_seq_abs_hip_quad_np = out_seq_quad_np.copy()
        last_x_out = initial_hip_x_pred
        last_z_out = initial_hip_z_pred
        for frame_idx in range(out_seq_abs_hip_quad_np.shape[0]):
            current_delta_x = out_seq_abs_hip_quad_np[frame_idx, 0]
            current_delta_z = out_seq_abs_hip_quad_np[frame_idx, 2]
            out_seq_abs_hip_quad_np[frame_idx, 0] = last_x_out + current_delta_x
            out_seq_abs_hip_quad_np[frame_idx, 2] = last_z_out + current_delta_z
            last_x_out = out_seq_abs_hip_quad_np[frame_idx, 0]
            last_z_out = out_seq_abs_hip_quad_np[frame_idx, 2]

        out_seq_euler_bvh = convert_quad_seq_to_euler_bvh(out_seq_abs_hip_quad_np, skeleton_info, ordered_bones_list, rc)
        if out_seq_euler_bvh.size > 0: # Check if conversion produced data
            read_bvh.write_traindata_to_bvh(standard_bvh_file, os.path.join(save_dance_folder, "%07d"%iteration+"_out.bvh"), out_seq_euler_bvh)
            print(f"Saved GT and Predicted BVH for iter {iteration} to {save_dance_folder}")
        else:
            print(f"Warning: Predicted Euler sequence for iteration {iteration} is empty after conversion. Skipping save.")
    elif save_bvh_motion==True and not (standard_bvh_file and skeleton_info and ordered_bones_list):
        print(f"Iteration {iteration}: Skipping BVH save because standard_bvh_file, skeleton_info, or ordered_bones_list is missing.")

def get_dance_len_lst(dances):
    len_lst=[]
    for dance in dances:
        length = max(1, dance.shape[0] // 100)
        len_lst.append(length)
    index_lst=[]
    current_idx=0
    for length in len_lst:
        for _ in range(length):
            index_lst.append(current_idx)
        current_idx += 1
    return index_lst

def load_dances(dance_folder):
    dance_files=os.listdir(dance_folder)
    dances=[]
    print('Loading motion files...')
    for dance_file in dance_files:
        if not dance_file.endswith(".npy"):
            continue
        dance_path = os.path.join(dance_folder, dance_file)
        try:
            dance=np.load(dance_path)
            dances.append(dance)
        except Exception as e:
            print(f"Could not load or process {dance_path}: {e}")
    print(len(dances), 'Motion files loaded.')
    if not dances:
        print(f"Warning: No dance files loaded from {dance_folder}.")
    return dances
    
def train(dances, frame_rate, batch_size_arg, seq_len_arg, read_weight_path, write_weight_folder,
          write_bvh_motion_folder, in_frame_arg, out_frame_arg, hidden_size_arg=1024, total_iter=100000,
          standard_bvh_file_arg=None): # Added standard_bvh_file_arg
    sample_seq_len = seq_len_arg + 2
    torch.cuda.set_device(0)
    model = acLSTM(in_frame_size=in_frame_arg, hidden_size=hidden_size_arg, out_frame_size=out_frame_arg)
    if read_weight_path and os.path.exists(read_weight_path):
        model.load_state_dict(torch.load(read_weight_path))
        print(f"Loaded weights from {read_weight_path}")
    elif read_weight_path:
        print(f"Weight file {read_weight_path} not found. Training from scratch.")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.cuda()
    model.train()
    if not dances:
        print("No dances to train on. Exiting.")
        return
    dance_len_lst=get_dance_len_lst(dances)
    if not dance_len_lst:
        print("Dance length list is empty. Exiting.")
        return
    random_range=len(dance_len_lst)
    speed=frame_rate/30.0

    # Load skeleton information
    skeleton = None
    ordered_bones_with_rotations = []
    if standard_bvh_file_arg and os.path.exists(standard_bvh_file_arg):
        print(f"Loading skeleton from: {standard_bvh_file_arg}")
        skeleton, non_end_bones = read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file_arg)
        # Create an ordered list of bones that have rotations, hip first, then others from non_end_bones
        # This order must match how generate_training_quad_data.py structured its quaternion data
        if skeleton:
            hip_name = 'hip' # Standard hip name, adjust if different in your BVH
            # Ensure 'hip' is processed first if it has rotations, then other non-end bones
            temp_ordered_bones = []
            if hip_name in skeleton and any('rotation' in ch.lower() for ch in skeleton[hip_name].get('channels',[])):
                temp_ordered_bones.append(hip_name)
            
            for bone in non_end_bones:
                if bone != hip_name and bone in skeleton and any('rotation' in ch.lower() for ch in skeleton[bone].get('channels',[])):
                    temp_ordered_bones.append(bone)
            ordered_bones_with_rotations = temp_ordered_bones
            print(f"Order of bones for rotation processing: {ordered_bones_with_rotations}")
            if not ordered_bones_with_rotations:
                print("Warning: No bones with rotations found in the skeleton from standard BVH. BVH saving might be incorrect.")
        else:
            print(f"Warning: Could not load skeleton from {standard_bvh_file_arg}. BVH saving will be incorrect.")
    else:
        print(f"Warning: Standard BVH file '{standard_bvh_file_arg}' not found or not provided. BVH saving will be incorrect.")

    for iteration in range(total_iter):
        dance_batch_item_np_lst=[] # Renamed from dance_batch for clarity within loop
        retries = 0
        max_retries = batch_size_arg * 5 # Allow more retries to find suitable dances
        b_idx = 0
        while b_idx < batch_size_arg and retries < max_retries:
            dance_id_idx = dance_len_lst[np.random.randint(0,random_range)]
            dance=dances[dance_id_idx].copy()
            dance_len_frames = dance.shape[0]
            required_raw_frames = int(sample_seq_len * speed) + 1

            if dance_len_frames < required_raw_frames + 20:
                retries +=1
                if retries >= max_retries and b_idx == 0: # If continuously failing for the first batch item
                    print("Failed to find any suitable long dance after many retries. Check data.")
                    return # or break / handle error
                continue # Try another dance for this batch item
            
            max_start_id = dance_len_frames - required_raw_frames - 10
            start_id=random.randint(10, max(10, max_start_id))
            current_sample_seq=[]
            for i in range(sample_seq_len):
                frame_to_sample = dance[int(i*speed+start_id)]
                current_sample_seq.append(frame_to_sample)
            dance_batch_item_np_lst.append(current_sample_seq)
            b_idx += 1
            retries = 0 # Reset retries for next batch item
        
        if len(dance_batch_item_np_lst) < batch_size_arg:
            print(f"Warning: Could only form a batch of size {len(dance_batch_item_np_lst)} due to short dances.")
            if not dance_batch_item_np_lst: 
                print("Skipping iteration due to empty batch.")
                continue
        
        real_seq_np = np.array(dance_batch_item_np_lst)
        
        # Determine save folder for this iteration's BVHs
        # Using a simpler iteration-based folder name to avoid complexity with dance_id_idx from inner loop
        iter_specific_bvh_save_folder = os.path.join(write_bvh_motion_folder, "%07d" % iteration)
        if not os.path.exists(iter_specific_bvh_save_folder):
            os.makedirs(iter_specific_bvh_save_folder)
        
        print_loss_flag = (iteration % 200 == 0)
        save_bvh_motion_flag = (iteration % 2000 == 0)

        if save_bvh_motion_flag and (not skeleton or not ordered_bones_with_rotations):
            print(f"Warning: Iteration {iteration}: Cannot save BVH, skeleton info missing. Forcing save_bvh_motion_flag to False.")
            current_save_bvh_flag = False
        else:
            current_save_bvh_flag = save_bvh_motion_flag

        train_one_iteraton(real_seq_np, model, optimizer, iteration, 
                           iter_specific_bvh_save_folder, # Pass the directory, train_one_iteraton will append filenames
                           print_loss_flag, 
                           current_save_bvh_flag, # Use potentially modified flag
                           standard_bvh_file=standard_bvh_file_arg,
                           skeleton_info=skeleton, 
                           ordered_bones_list=ordered_bones_with_rotations)

        if iteration % 10000 == 0 and iteration > 0:
            if not os.path.exists(write_weight_folder):
                os.makedirs(write_weight_folder)
            path = os.path.join(write_weight_folder, "%07d"%iteration +".weight")
            torch.save(model.state_dict(), path)

    if not os.path.exists(write_weight_folder+"finish/"):
        os.makedirs(write_weight_folder+"finish/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dances_folder', type=str, required=True, help='Path for the training data (npy files)')
    parser.add_argument('--write_weight_folder', type=str, required=True, help='Path to store checkpoints')
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True, help='Path to store test generated bvh during training')
    parser.add_argument('--read_weight_path', type=str, default="", help='Path to pre-trained checkpoint model path')
    parser.add_argument('--dance_frame_rate', type=int, default=60, help='Frame rate of the source BVH data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--in_frame', type=int, required=True, help='Input channels per frame (e.g., 3 + num_joints_with_rot * 4 for quat)')
    parser.add_argument('--out_frame', type=int, required=True, help='Output channels per frame (must match in_frame)')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size of LSTM layers')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length for LSTM processing')
    parser.add_argument('--total_iterations', type=int, default=100000, help='Total training iterations')
    parser.add_argument('--standard_bvh_file', type=str, default='train_data_bvh/standard.bvh', help='Path to the standard BVH file for skeleton hierarchy.')
    args = parser.parse_args()

    for folder_path in [args.write_weight_folder, args.write_bvh_motion_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    if not os.path.exists(args.dances_folder):
        print(f"ERROR: Dances folder {args.dances_folder} does not exist!")
        return

    dances_data = load_dances(args.dances_folder)
    if not dances_data:
        print(f"No data loaded from {args.dances_folder}. Exiting.")
        return

    train(dances_data, args.dance_frame_rate, args.batch_size, args.seq_len, 
          args.read_weight_path, args.write_weight_folder, args.write_bvh_motion_folder,
          args.in_frame, args.out_frame, args.hidden_size, total_iter=args.total_iterations,
          standard_bvh_file_arg=args.standard_bvh_file)

if __name__ == '__main__':
    main()
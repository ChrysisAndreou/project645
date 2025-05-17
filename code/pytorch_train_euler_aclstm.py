import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh # Assuming read_bvh.py is in the same directory or accessible
import argparse

# Defined in generate_training_euler_data.py, ensure consistency
HIP_POS_SCALE_FACTOR = 0.01 

# Hip_index for accessing X, Y, Z position channels directly (0, 1, 2)
HIP_X_CHANNEL, HIP_Y_CHANNEL, HIP_Z_CHANNEL = 0, 1, 2

# Global parameters (can be overridden by args)
Condition_num = 5
Groundtruth_num = 5

class acLSTM(nn.Module):
    def __init__(self, in_frame_size, hidden_size, out_frame_size):
        super(acLSTM, self).__init__()
        
        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size
        
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
    
    def init_hidden(self, batch_size):
        c0 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).cuda())
        c1 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).cuda())
        c2 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).cuda())
        h0 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).cuda())
        h1 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).cuda())
        h2 = Variable(torch.FloatTensor(np.zeros((batch_size, self.hidden_size))).cuda())
        return ([h0,h1,h2], [c0,c1,c2])
    
    def forward_lstm(self, in_frame, vec_h, vec_c):
        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h0, (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h1, (vec_h[2], vec_c[2]))
        out_frame = self.decoder(vec_h2)
        return (out_frame, [vec_h0, vec_h1, vec_h2], [vec_c0, vec_c1, vec_c2])
        
    def get_condition_lst(self, condition_num, groundtruth_num, seq_len):
        # Ensure enough elements for at least one full cycle if seq_len is small
        num_cycles = seq_len // (condition_num + groundtruth_num) + 1 
        gt_lst = np.ones((num_cycles, groundtruth_num))
        con_lst = np.zeros((num_cycles, condition_num))
        # Interleave ground truth and conditioned frames
        lst = np.empty((num_cycles, condition_num + groundtruth_num), dtype=gt_lst.dtype)
        lst[:, :groundtruth_num] = gt_lst
        lst[:, groundtruth_num:] = con_lst
        return lst.reshape(-1)[:seq_len]
        
    def forward(self, real_seq, condition_num, groundtruth_num):
        batch_size = real_seq.size()[0]
        seq_len = real_seq.size()[1]
        
        condition_lst = self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        (vec_h, vec_c) = self.init_hidden(batch_size)
        
        out_seq_list = []
        # Initialize out_frame for the first conditioned step if seq starts with it
        out_frame = Variable(torch.FloatTensor(np.zeros((batch_size, self.out_frame_size))).cuda())
        
        for i in range(seq_len):
            if condition_lst[i] == 1:
                in_frame = real_seq[:,i]
            else:
                in_frame = out_frame # Use previously generated frame
            
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq_list.append(out_frame.unsqueeze(1))
        
        return torch.cat(out_seq_list, dim=1) # Shape: (batch, seq_len, frame_size)
    
    def calculate_loss(self, pred_seq, groundtruth_seq):
        # pred_seq and groundtruth_seq are [batch_size, seq_len, self.out_frame_size]
        # Data is already scaled and relative (for X,Z hip) or scaled absolute (Y hip)
        
        hip_pos_pred = pred_seq[..., HIP_X_CHANNEL:HIP_Z_CHANNEL+1] # X, Y, Z
        hip_pos_gt = groundtruth_seq[..., HIP_X_CHANNEL:HIP_Z_CHANNEL+1]
        
        rot_pred_rad = pred_seq[..., HIP_Z_CHANNEL+1:] # Rotations start after hip Z
        rot_gt_rad = groundtruth_seq[..., HIP_Z_CHANNEL+1:]

        loss_fn_mse = nn.MSELoss()
        loss_hip = loss_fn_mse(hip_pos_pred, hip_pos_gt)

        cos_error = torch.cos(rot_pred_rad - rot_gt_rad)
        loss_rot_ad = torch.mean(1.0 - cos_error)
        
        total_loss = loss_hip + loss_rot_ad
        return total_loss

def train_one_iteration(real_seq_np, model, optimizer, iteration, save_dance_folder, 
                        print_loss=False, save_bvh_motion=False, standard_bvh_ref_path=None):
    # real_seq_np is [batch, sampling_seq_len, frame_size]
    # Hip X,Z are relative to their original first frame & scaled. Hip Y is scaled absolute.
    # Angles are in radians.

    # Calculate frame-to-frame differences for hip X and Z of the SCALED data.
    # Hip Y and all rotations remain as they are (Y absolute scaled, rotations absolute radians).
    dif_hip_x_scaled = real_seq_np[:, 1:, HIP_X_CHANNEL] - real_seq_np[:, :-1, HIP_X_CHANNEL]
    dif_hip_z_scaled = real_seq_np[:, 1:, HIP_Z_CHANNEL] - real_seq_np[:, :-1, HIP_Z_CHANNEL]
    
    # Prepare sequence for LSTM: use N-1 frames, with X,Z diffs
    # Target sequence will be the N-1 frames, offset by 1 from input sequence
    real_seq_processed_for_lstm_np = real_seq_np[:, :-1].copy() 
    real_seq_processed_for_lstm_np[:, :, HIP_X_CHANNEL] = dif_hip_x_scaled
    real_seq_processed_for_lstm_np[:, :, HIP_Z_CHANNEL] = dif_hip_z_scaled

    real_seq_cuda = Variable(torch.FloatTensor(real_seq_processed_for_lstm_np).cuda())
 
    # Input to model: first `args.seq_len` frames of the processed sequence.
    # Target for model: next `args.seq_len` frames (offset by 1) of the processed sequence.
    in_lstm_seq = real_seq_cuda[:, :-1]      
    target_lstm_seq = real_seq_cuda[:, 1:] 
    
    optimizer.zero_grad()
    pred_seq = model.forward(in_lstm_seq, Condition_num, Groundtruth_num) 
    loss = model.calculate_loss(pred_seq, target_lstm_seq)
    loss.backward()
    optimizer.step()
    
    if print_loss:
        print(f"########### iter {iteration:07d} ######################")
        print(f"loss: {loss.item():.4f}") # Loss is on scaled data
    
    if save_bvh_motion and standard_bvh_ref_path:
        # Get first sequence from batch for saving
        # target_lstm_seq and pred_seq are already on CPU via .data.cpu().numpy()
        gt_to_save_scaled_np = target_lstm_seq[0].data.cpu().numpy()
        pred_to_save_scaled_np = pred_seq[0].data.cpu().numpy()

        # For BVH, reconstruct hip positions from scaled differences and inverse scale
        # Rotations are already in radians, just need to convert to degrees.
        for seq_idx, scaled_np_data in enumerate([gt_to_save_scaled_np, pred_to_save_scaled_np]):
            # This data has scaled X,Z differences and scaled absolute Y.
            # Create a copy for BVH with unscaled, absolute positions.
            bvh_ready_data = scaled_np_data.copy()
            
            # Accumulate scaled X,Z differences to get scaled absolute X,Z path
            # Start accumulation from (0,0) for this segment as we don't have prev frame's absolute here.
            current_abs_scaled_x = 0.0 
            current_abs_scaled_z = 0.0
            for frame_idx in range(bvh_ready_data.shape[0]):
                # X and Z in bvh_ready_data are diffs, make them absolute (still scaled)
                bvh_ready_data[frame_idx, HIP_X_CHANNEL] += current_abs_scaled_x
                current_abs_scaled_x = bvh_ready_data[frame_idx, HIP_X_CHANNEL]
                
                bvh_ready_data[frame_idx, HIP_Z_CHANNEL] += current_abs_scaled_z
                current_abs_scaled_z = bvh_ready_data[frame_idx, HIP_Z_CHANNEL]
            
            # Inverse scale hip positions
            bvh_ready_data[:, HIP_X_CHANNEL] /= HIP_POS_SCALE_FACTOR
            bvh_ready_data[:, HIP_Y_CHANNEL] /= HIP_POS_SCALE_FACTOR # Y was absolute scaled
            bvh_ready_data[:, HIP_Z_CHANNEL] /= HIP_POS_SCALE_FACTOR
            
            # Convert rotations to degrees
            bvh_ready_data[:, HIP_Z_CHANNEL+1:] = np.rad2deg(bvh_ready_data[:, HIP_Z_CHANNEL+1:])

            file_suffix = "_gt.bvh" if seq_idx == 0 else "_out.bvh"
            read_bvh.write_frames(standard_bvh_ref_path, 
                                  os.path.join(save_dance_folder, f"{iteration:07d}{file_suffix}"), 
                                  bvh_ready_data)

def get_dance_len_lst(dances):
    len_lst = []
    for dance in dances:
        length = len(dance) // 100 # Approximation for weighting, can be tuned
        len_lst.append(max(1, length)) # Ensure at least 1 count
    
    index_lst = []
    for i, length in enumerate(len_lst):
        index_lst.extend([i] * length)
    return index_lst

def load_dances(dance_folder):
    dance_files = [f for f in os.listdir(dance_folder) if f.endswith(".npy")]
    dances = []
    print(f'Loading motion files from {dance_folder}...')
    for dance_file in dance_files:
        try:
            dance = np.load(os.path.join(dance_folder, dance_file))
            if dance.ndim == 2 and dance.shape[0] > 0 and dance.shape[1] > 0: # Basic check
                dances.append(dance)
            else:
                print(f"Warning: Skipping invalid or empty dance file: {dance_file}")
        except Exception as e:
            print(f"Warning: Could not load dance file {dance_file}: {e}")
    print(f'{len(dances)} motion files loaded.')
    return dances
    
def train(dances, args):
    sampling_seq_len = args.seq_len + 2 
    if torch.cuda.is_available():
        torch.cuda.set_device(0) # Or use args.gpu_id
        print("Training on GPU: cuda:0")
    else:
        print("Training on CPU")

    if not dances:
        print("Error: No dance data loaded. Cannot determine frame size.")
        return
    
    actual_frame_size = dances[0].shape[1]
    current_in_frame_size = actual_frame_size
    current_out_frame_size = actual_frame_size 

    if args.in_frame != -1 and args.in_frame != current_in_frame_size:
        print(f"Warning: Arg in_frame ({args.in_frame}) differs from data ({current_in_frame_size}). Using data's.")
    if args.out_frame != -1 and args.out_frame != current_out_frame_size:
        print(f"Warning: Arg out_frame ({args.out_frame}) differs from data ({current_out_frame_size}). Using data's.")

    model = acLSTM(in_frame_size=current_in_frame_size, hidden_size=args.hidden_size, out_frame_size=current_out_frame_size)
    
    if args.read_weight_path:
        if os.path.exists(args.read_weight_path):
            print(f"Loading weights from: {args.read_weight_path}")
            model.load_state_dict(torch.load(args.read_weight_path))
        else:
            print(f"Warning: Weight path {args.read_weight_path} not found. Starting from scratch.")
    
    if torch.cuda.is_available(): model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    
    dance_len_lst = get_dance_len_lst(dances)
    if not dance_len_lst:
        print("Error: No dances to train on after processing lengths.")
        return
    random_range = len(dance_len_lst)
    
    speed = args.dance_frame_rate / 30.0 
    
    for iteration in range(args.total_iterations):
        dance_batch_list = []
        for _ in range(args.batch_size):
            dance_id = dance_len_lst[np.random.randint(0, random_range)]
            dance = dances[dance_id]
            dance_len = dance.shape[0]
            
            min_req_len = int(sampling_seq_len * speed) + 20 # +20 for safety margin and start_id offset
            if dance_len < min_req_len:
                # Simple fallback: repeat last frame if too short
                # print(f"Warning: Dance {dance_id} too short ({dance_len} frames). Required {min_req_len}. Padding.")
                needed_frames_at_speed = int(sampling_seq_len * speed)
                indices = (np.arange(needed_frames_at_speed) * speed).astype(int)
                indices = np.clip(indices, 0, dance_len - 1)
                sample_seq = dance[indices[:sampling_seq_len]] # Ensure we take exactly sampling_seq_len
                if sample_seq.shape[0] < sampling_seq_len:
                    padding_count = sampling_seq_len - sample_seq.shape[0]
                    padding = np.tile(sample_seq[-1], (padding_count, 1))
                    sample_seq = np.vstack((sample_seq, padding))
            else:
                start_id_max = dance_len - int(sampling_seq_len * speed) - 10
                start_id = random.randint(10, start_id_max if start_id_max >=10 else 10)
                sample_indices = (np.arange(sampling_seq_len) * speed + start_id).astype(int)
                sample_seq = dance[sample_indices]

            # Data Augmentation (Simplified: only hip translation, as original was for positional)
            # Data is already scaled.
            aug_T_scaled = [0.1*(random.random()-0.5)*HIP_POS_SCALE_FACTOR, 
                            0.0,  # No Y augmentation to keep ground contact sensible
                            0.1*(random.random()-0.5)*HIP_POS_SCALE_FACTOR]
            
            sample_seq_augmented = sample_seq.copy()
            sample_seq_augmented[:, HIP_X_CHANNEL] += aug_T_scaled[0]
            # sample_seq_augmented[:, HIP_Y_CHANNEL] += aug_T_scaled[1] # Typically Y is not augmented this way
            sample_seq_augmented[:, HIP_Z_CHANNEL] += aug_T_scaled[2]
            
            dance_batch_list.append(sample_seq_augmented)
        
        dance_batch_np = np.array(dance_batch_list)
       
        print_loss_flag = (iteration % 20 == 0)
        save_bvh_flag = (iteration % 1000 == 0) and args.standard_bvh_reference
        
        train_one_iteration(dance_batch_np, model, optimizer, iteration, 
                              args.write_bvh_motion_folder, 
                              print_loss=print_loss_flag, save_bvh_motion=save_bvh_flag,
                              standard_bvh_ref_path=args.standard_bvh_reference)

        if iteration % 1000 == 0 and iteration > 0: # Save model checkpoint
            path = os.path.join(args.write_weight_folder, f"{iteration:07d}.weight")
            torch.save(model.state_dict(), path)
            print(f"Saved model weights to {path}")

def main():
    parser = argparse.ArgumentParser(description="Train ACLSTM for Euler Angle Motion with Normalized Hip")
    parser.add_argument('--dances_folder', type=str, required=True, help='Path for normalized training data (.npy)')
    parser.add_argument('--write_weight_folder', type=str, required=True, help='Path to store model checkpoints')
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True, help='Path to store sample generated BVH')
    parser.add_argument('--standard_bvh_reference', type=str, help='Path to standard BVH for hierarchy (required for saving BVH)')
    parser.add_argument('--read_weight_path', type=str, default="", help='Path to pre-trained model to continue training')
    
    parser.add_argument('--dance_frame_rate', type=int, default=60, help='Frame rate of source BVH data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--in_frame', type=int, default=-1, help='Input frame size. -1 to auto-detect.')
    parser.add_argument('--out_frame', type=int, default=-1, help='Output frame size. -1 to auto-detect.')
    parser.add_argument('--hidden_size', type=int, default=1024, help='LSTM hidden size')
    parser.add_argument('--seq_len', type=int, default=100, help='LSTM sequence length for training')
    parser.add_argument('--total_iterations', type=int, default=50000, help='Total training iterations')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')

    args = parser.parse_args()

    os.makedirs(args.write_weight_folder, exist_ok=True)
    os.makedirs(args.write_bvh_motion_folder, exist_ok=True)
    
    if args.standard_bvh_reference and not os.path.exists(args.standard_bvh_reference):
        print(f"Error: Standard BVH reference not found at {args.standard_bvh_reference}. BVH saving will fail.")
        # Potentially exit or disable BVH saving if critical

    dances_data = load_dances(args.dances_folder)
    if not dances_data:
        print(f"No data found in {args.dances_folder}. Exiting.")
        return

    train(dances_data, args)

if __name__ == '__main__':
    main()

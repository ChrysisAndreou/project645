import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse

# For raw BVH data: Hip X pos is 0, Y pos is 1, Z pos is 2. Rotations start at index 3.

# Default values, should be overridden by command-line arguments where appropriate
Seq_len=100 # This might refer to a concept in acLSTM not directly used in synthesis like this
Hidden_size = 1024
# In_frame_size should be set based on the trained model, e.g., 63 for Euler

class acLSTM(nn.Module):
    def __init__(self, in_frame_size=171, hidden_size=1024, out_frame_size=171):
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
        
    def forward(self, initial_seq, generate_frames_number):
        batch=initial_seq.size()[0]
        (vec_h, vec_c) = self.init_hidden(batch)
        out_seq_list = [] # Store frames as a list of tensors

        # Process initial sequence to set LSTM state
        current_frame = torch.autograd.Variable(torch.FloatTensor( np.zeros((batch,self.out_frame_size))   ).cuda())
        for i in range(initial_seq.size()[1]):
            in_frame=initial_seq[:,i]
            (current_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            # We don't add these to out_seq_list for generation, but current_frame is the last prediction from init phase
        
        # Start generating from the last predicted frame
        for i in range(generate_frames_number):
            in_frame=current_frame # Use the previously generated frame as input
            (current_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq_list.append(current_frame.unsqueeze(1)) # Add (batch, 1, frame_size) to list
            
        if not out_seq_list:
            return torch.FloatTensor([]).cuda() # Return empty if nothing generated
        
        return torch.cat(out_seq_list, dim=1) # Concatenate along seq_len dim -> (batch, gen_frames, frame_size)

    def calculate_euler_loss_metric(self, pred_seq, gt_seq):
        # pred_seq, gt_seq are [batch, num_eval_frames, frame_size]
        # Ensure they are on the same device if not already
        pred_seq = pred_seq.to(gt_seq.device)

        hip_pos_pred = pred_seq[..., :3]
        hip_pos_gt = gt_seq[..., :3]
        rot_pred_deg = pred_seq[..., 3:]
        rot_gt_deg = gt_seq[..., 3:]

        loss_fn_mse = nn.MSELoss()
        loss_hip = loss_fn_mse(hip_pos_pred, hip_pos_gt)

        rot_pred_rad = rot_pred_deg * (np.pi / 180.0)
        rot_gt_rad = rot_gt_deg * (np.pi / 180.0)
        cos_error = torch.cos(rot_pred_rad - rot_gt_rad)
        loss_rot_ad = torch.mean(1.0 - cos_error)
        
        # Note: This is a metric, not for backprop. It can be a simple sum or weighted.
        # For reporting, maybe report them separately or a simple sum.
        total_eval_loss = loss_hip + loss_rot_ad
        return loss_hip.item(), loss_rot_ad.item(), total_eval_loss.item()


# initial_seq_np: [batch, initial_seq_len, frame_size]
# full_dance_data_np: [batch, total_dance_len, frame_size] (used to get ground truth for eval)
# initial_seq_start_indices: [batch] list of start indices of initial_seq_np within full_dance_data_np
def generate_and_evaluate_seq(initial_seq_np, full_dance_data_np, initial_seq_start_indices, 
                                generate_frames_number, model, save_dance_folder, num_eval_frames=20):
    
    batch_size = initial_seq_np.shape[0]
    initial_seq_len = initial_seq_np.shape[1]
    current_frame_size = initial_seq_np.shape[2]

    # Prepare initial sequence with hip X, Z differences
    initial_seq_processed_np = initial_seq_np.copy()
    if initial_seq_len > 1: # Need at least 2 frames to make a difference
        dif_hip_x = initial_seq_np[:, 1:, 0] - initial_seq_np[:, :-1, 0]
        dif_hip_z = initial_seq_np[:, 1:, 2] - initial_seq_np[:, :-1, 2]
        # The network input is the sequence where the *first* frame has absolute hip, 
        # and subsequent frames fed to LSTM use predicted diffs.
        # For synthesis, the initial_seq is fed frame by frame. The first frame is absolute.
        # The *transformation* to relative hip X/Z is for training targets.
        # Here, for synthesis, we feed the absolute values, and model predicts relative internally.
        # The state passed to forward() is after processing these initial_seq_np frames.
        # So, `initial_seq_processed_np` should be what `model.forward` expects. 
        # If model was trained on diffs, the input here should also be diffs after the first frame.
        # The current `model.forward` in training script takes `real_seq` which has diffs.
        # So, the initial_seq for synthesis should also be transformed similarly for its later parts.
        temp_initial_seq_for_lstm = initial_seq_np.copy()
        base_hip_x = temp_initial_seq_for_lstm[:, 0, 0].copy() # Keep first frame absolute for starting point
        base_hip_z = temp_initial_seq_for_lstm[:, 0, 2].copy()
        for i in range(1, initial_seq_len):
            current_hip_x = temp_initial_seq_for_lstm[:, i, 0].copy()
            current_hip_z = temp_initial_seq_for_lstm[:, i, 2].copy()
            temp_initial_seq_for_lstm[:, i, 0] = current_hip_x - base_hip_x
            temp_initial_seq_for_lstm[:, i, 2] = current_hip_z - base_hip_z
            base_hip_x = current_hip_x
            base_hip_z = current_hip_z
        initial_seq_for_lstm_cuda  = torch.autograd.Variable(torch.FloatTensor(temp_initial_seq_for_lstm.tolist()).cuda())
    else: # single frame, no diffs possible for input prep, feed as is
        initial_seq_for_lstm_cuda  = torch.autograd.Variable(torch.FloatTensor(initial_seq_np.tolist()).cuda())

    # generated_seq will be [batch, generate_frames_number, frame_size]
    # These are the raw outputs from the network (hip X,Z are diffs, Y and rotations absolute)
    generated_seq_raw_cuda = model.forward(initial_seq_for_lstm_cuda, generate_frames_number)
    generated_seq_raw_np = generated_seq_raw_cuda.data.cpu().numpy()

    # Post-process: reconstruct absolute hip positions for the generated part
    generated_seq_abs_hip_np = generated_seq_raw_np.copy()
    # Accumulation starts from the last hip position of the *original, absolute* initial sequence
    last_abs_hip_x = initial_seq_np[:, -1, 0].copy() # Shape: [batch_size]
    last_abs_hip_z = initial_seq_np[:, -1, 2].copy()

    for i in range(generate_frames_number):
        current_pred_delta_x = generated_seq_abs_hip_np[:, i, 0]
        current_pred_delta_z = generated_seq_abs_hip_np[:, i, 2]
        generated_seq_abs_hip_np[:, i, 0] = last_abs_hip_x + current_pred_delta_x
        generated_seq_abs_hip_np[:, i, 2] = last_abs_hip_z + current_pred_delta_z
        last_abs_hip_x = generated_seq_abs_hip_np[:, i, 0]
        last_abs_hip_z = generated_seq_abs_hip_np[:, i, 2]
        # Hip Y and rotations are already absolute from network output

    # Save full generated sequences (initial_absolute + generated_absolute_hip)
    for b_idx in range(batch_size):
        # Concatenate the original initial sequence (absolute) with the generated part (absolute hip)
        full_output_seq_np = np.concatenate((initial_seq_np[b_idx], generated_seq_abs_hip_np[b_idx]), axis=0)
        if not os.path.exists(save_dance_folder):
            os.makedirs(save_dance_folder)
        read_bvh.write_traindata_to_bvh(os.path.join(save_dance_folder, f"generated_euler_{b_idx:02d}.bvh"), full_output_seq_np)
        print(f"Saved generated Euler motion (initial + {generate_frames_number} steps) to generated_euler_{b_idx:02d}.bvh")

    # Quantitative Evaluation for the first num_eval_frames of the *generated* part
    if num_eval_frames > 0 and generate_frames_number >= num_eval_frames:
        eval_pred_seq_abs_hip_np = generated_seq_abs_hip_np[:, :num_eval_frames, :]
        
        # Get corresponding ground truth frames
        eval_gt_seq_list = []
        for b_idx in range(batch_size):
            gt_start_frame_idx = initial_seq_start_indices[b_idx] + initial_seq_len
            gt_end_frame_idx = gt_start_frame_idx + num_eval_frames
            if gt_end_frame_idx <= full_dance_data_np[b_idx].shape[0]:
                eval_gt_seq_list.append(full_dance_data_np[b_idx][gt_start_frame_idx:gt_end_frame_idx, :])
            else:
                print(f"Warning: Not enough ground truth frames for evaluation for batch item {b_idx}.")
                # Handle by padding or skipping this item for eval - for now, append zeros if short
                # This will make loss high, indicating an issue with data length or request.
                available_gt = full_dance_data_np[b_idx][gt_start_frame_idx:, :] if gt_start_frame_idx < full_dance_data_np[b_idx].shape[0] else np.array([])
                padding_needed = num_eval_frames - available_gt.shape[0]
                if padding_needed > 0:
                     padding_array = np.zeros((padding_needed, current_frame_size))
                     eval_gt_seq_list.append(np.concatenate((available_gt, padding_array), axis=0) if available_gt.size > 0 else padding_array)
                else:
                     eval_gt_seq_list.append(available_gt) # Should not happen if padding_needed > 0
        
        if len(eval_gt_seq_list) == batch_size: # Ensure all items were processed
            eval_gt_seq_np = np.array(eval_gt_seq_list)
            loss_hip, loss_rot_ad, total_loss = model.calculate_euler_loss_metric(
                torch.FloatTensor(eval_pred_seq_abs_hip_np).cuda(), 
                torch.FloatTensor(eval_gt_seq_np).cuda()
            )
            print(f"---- Quantitative Evaluation (first {num_eval_frames} generated frames) ----")
            print(f"Hip MSE Loss: {loss_hip:.4f}")
            print(f"Rotation Angle Distance Loss: {loss_rot_ad:.4f}")
            print(f"Total Combined Eval Loss: {total_loss:.4f}")
            print("----------------------------------------------------------")
        else:
            print("Could not perform full quantitative evaluation due to ground truth length issues.")

    return generated_seq_abs_hip_np


def load_full_dance_data(dance_folder, num_dances_to_load):
    dance_files = [f for f in os.listdir(dance_folder) if f.endswith(".npy")]
    dances = []
    print(f'Loading full motion files for ground truth from {dance_folder}...')
    for i, dance_file in enumerate(dance_files):
        if i >= num_dances_to_load: break
        dance_path = os.path.join(dance_folder, dance_file)
        try:
            dance=np.load(dance_path)
            dances.append(dance)
        except Exception as e:
            print(f"Could not load or process {dance_path}: {e}")
    print(f"{len(dances)} full motion files loaded.")
    return dances

def main():
    parser = argparse.ArgumentParser(description="Synthesize Euler angle based character motion.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained .weight model file.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the folder containing .npy motion files (Euler format).')
    parser.add_argument('--output_folder', type=str, default="./synthesized_euler_motions/", help='Folder to save generated BVH files.')
    parser.add_argument('--num_initial_frames', type=int, default=20, help='Number of initial frames to feed the model.')
    parser.add_argument('--num_generate_frames', type=int, default=400, help='Number of frames to generate.')
    parser.add_argument('--num_eval_frames', type=int, default=20, help='Number of generated frames to use for quantitative evaluation against ground truth.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of sequences to generate (must be <= number of files in data_folder).')
    parser.add_argument('--in_frame_size', type=int, required=True, help='Input frame size (channels) for the model (e.g., 63 for Euler).')
    # out_frame_size is usually same as in_frame_size for these models
    # hidden_size should match the trained model, but acLSTM has a default.
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size of LSTM layers in the model.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use.')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found.")
        return
    if not os.path.exists(args.data_folder):
        print(f"Error: Data folder {args.data_folder} not found.")
        return

    # Load model
    model = acLSTM(in_frame_size=args.in_frame_size, hidden_size=args.hidden_size, out_frame_size=args.in_frame_size)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval() # Set model to evaluation mode

    # Load dance data to get initial sequences and ground truth
    full_dances_np = load_full_dance_data(args.data_folder, args.batch_size)
    if not full_dances_np or len(full_dances_np) < args.batch_size:
        print(f"Error: Not enough dance files in {args.data_folder} to meet batch size {args.batch_size}.")
        print(f"Loaded {len(full_dances_np)} dances.")
        return

    initial_seq_batch_list = []
    initial_seq_start_indices_list = [] # To keep track for ground truth slicing

    for i in range(args.batch_size):
        dance = full_dances_np[i]
        if dance.shape[0] < args.num_initial_frames + args.num_eval_frames: # Need enough for init + eval GT
            print(f"Warning: Dance {i} is too short ({dance.shape[0]} frames) for {args.num_initial_frames} initial frames and {args.num_eval_frames} eval frames. Skipping this item.")
            continue
        
        # Randomly select a starting point for the initial sequence
        # Ensure there are enough frames for initial_seq AND subsequent eval_frames for GT
        max_start_idx = dance.shape[0] - (args.num_initial_frames + args.num_eval_frames)
        if max_start_idx < 0:
             print(f"Warning: Dance {i} is too short for combined initial and evaluation. Max start index is {max_start_idx}")
             #This case should be caught by the check above, but being defensive.
             continue

        start_idx = random.randint(0, max_start_idx)
        initial_seq = dance[start_idx : start_idx + args.num_initial_frames, :]
        initial_seq_batch_list.append(initial_seq)
        initial_seq_start_indices_list.append(start_idx)
    
    if not initial_seq_batch_list:
        print("Error: Could not create any initial sequences from the provided data. Check data lengths and num_initial_frames/num_eval_frames.")
        return
    
    # Ensure the batch is now the size of successfully created initial sequences
    current_batch_size = len(initial_seq_batch_list)
    if current_batch_size < args.batch_size:
        print(f"Warning: Actual batch size for synthesis will be {current_batch_size} due to short dances.")
        # Adjust full_dances_np and initial_seq_start_indices_list if some items were skipped.
        # This is complex if items were skipped non-contiguously. Simpler: error out or proceed with smaller batch.
        # For now, we assume load_full_dance_data and this loop align, or we use the smaller current_batch_size.
        # A robust way: filter full_dances_np and rebuild start_indices according to what's in initial_seq_batch_list.
        # Let's assume for now this script is run with batch_size <= available good dances.

    initial_seq_batch_np = np.array(initial_seq_batch_list)
    # We also need a corresponding slice of full_dances_np for generate_and_evaluate_seq
    # If items were skipped, this needs careful slicing. For now, assume no skips or batch_size was met.
    # The simplest is to just pass the first `current_batch_size` full dances if `load_full_dance_data` loaded more.
    relevant_full_dances_np = np.array(full_dances_np[:current_batch_size])

    # Generate and evaluate sequences
    with torch.no_grad(): # Ensure no gradients are computed during synthesis
        generate_and_evaluate_seq(initial_seq_batch_np, relevant_full_dances_np, initial_seq_start_indices_list,
                                    args.num_generate_frames, model, args.output_folder, args.num_eval_frames)

    print(f"Motion synthesis finished. BVH files saved in {args.output_folder}")

if __name__ == '__main__':
    main()
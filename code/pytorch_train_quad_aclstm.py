import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse
import torch.nn.functional as F # For normalization

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


def train_one_iteraton(real_seq_np, model, optimizer, iteration, save_dance_folder, print_loss=False, save_bvh_motion=True):
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
        gt_seq_to_save = np.array(predict_groundtruth_seq_targets[0].data.cpu().numpy())
        initial_hip_x = real_seq_np[0, 0, 0]
        initial_hip_z = real_seq_np[0, 0, 2]
        gt_seq_bvh = gt_seq_to_save.copy()
        last_x_gt = initial_hip_x
        last_z_gt = initial_hip_z
        for frame_idx in range(gt_seq_bvh.shape[0]):
            current_delta_x = gt_seq_bvh[frame_idx, 0]
            current_delta_z = gt_seq_bvh[frame_idx, 2]
            gt_seq_bvh[frame_idx, 0] = last_x_gt + current_delta_x
            gt_seq_bvh[frame_idx, 2] = last_z_gt + current_delta_z
            last_x_gt = gt_seq_bvh[frame_idx, 0]
            last_z_gt = gt_seq_bvh[frame_idx, 2]
        out_seq_reshaped = np.array(predict_seq[0].data.cpu().numpy()).reshape(-1, current_frame_size)
        out_seq_bvh = out_seq_reshaped.copy()
        last_x_out = initial_hip_x
        last_z_out = initial_hip_z
        for frame_idx in range(out_seq_bvh.shape[0]):
            current_delta_x = out_seq_bvh[frame_idx, 0]
            current_delta_z = out_seq_bvh[frame_idx, 2]
            out_seq_bvh[frame_idx, 0] = last_x_out + current_delta_x
            out_seq_bvh[frame_idx, 2] = last_z_out + current_delta_z
            last_x_out = out_seq_bvh[frame_idx, 0]
            last_z_out = out_seq_bvh[frame_idx, 2]
        
        # NOTE: read_bvh.write_traindata_to_bvh expects Euler data for BVH conversion.
        # If gt_seq_bvh and out_seq_bvh are in Quaternion format (hip_pos + quats),
        # they need to be converted to Euler format before calling write_traindata_to_bvh.
        # This conversion step is missing here and would require generate_training_quad_data.py's
        # decoding logic (quad_to_euler part of generate_bvh_from_quad_traindata) or similar.
        # For now, this will save incorrect BVHs if data is quaternion.
        # This script should use a quaternion-to-BVH utility or save as .npy and use generate_training_quad_data.py for BVH.
        print("WARNING: Saving BVH from quaternion data is not fully implemented. BVHs might be incorrect.")
        print("Data saved below is likely hip_pos + quaternion rotations, not Euler for BVH.")
        # Option 1: Save as npy
        # np.save(os.path.join(save_dance_folder, "%07d"%iteration+"_gt_quad.npy"), gt_seq_bvh)
        # np.save(os.path.join(save_dance_folder, "%07d"%iteration+"_out_quad.npy"), out_seq_bvh)
        # Option 2: (Placeholder) call a quad_to_bvh writer if available
        # quad_to_bvh.write_quad_data_to_bvh(os.path.join(save_dance_folder, "%07d"%iteration+"_gt.bvh"), gt_seq_bvh, standard_bvh_file_for_header_if_needed)
        # quad_to_bvh.write_quad_data_to_bvh(os.path.join(save_dance_folder, "%07d"%iteration+"_out.bvh"), out_seq_bvh, standard_bvh_file_for_header_if_needed)
        # Fallback to current (incorrect for BVH visualization of quats)
        read_bvh.write_traindata_to_bvh(save_dance_folder+"%07d"%iteration+"_gt_rawquad.bvh", gt_seq_bvh) # Indicate it's raw
        read_bvh.write_traindata_to_bvh(save_dance_folder+"%07d"%iteration+"_out_rawquad.bvh", out_seq_bvh)

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
          write_bvh_motion_folder, in_frame_arg, out_frame_arg, hidden_size_arg=1024, total_iter=100000):
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

    for iteration in range(total_iter):
        dance_batch=[]
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
            dance_batch.append(current_sample_seq)
            b_idx += 1
            retries = 0 # Reset retries for next batch item
        
        if len(dance_batch) < batch_size_arg:
            print(f"Warning: Could only form a batch of size {len(dance_batch)} due to short dances.")
            if not dance_batch: 
                print("Skipping iteration due to empty batch.")
                continue
        
        dance_batch_np=np.array(dance_batch)
        print_loss_flag=(iteration % 20==0)
        save_bvh_motion_flag=(iteration % 1000==0 or iteration == total_iter -1)
        if save_bvh_motion_flag and not os.path.exists(write_weight_folder):
            os.makedirs(write_weight_folder)
        if save_bvh_motion_flag:
             path = os.path.join(write_weight_folder, "%07d"%iteration +".weight")
             torch.save(model.state_dict(), path)
        train_one_iteraton(dance_batch_np, model, optimizer, iteration, write_bvh_motion_folder, print_loss_flag, save_bvh_motion_flag)

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
          args.in_frame, args.out_frame, args.hidden_size, total_iter=args.total_iterations)

if __name__ == '__main__':
    main()
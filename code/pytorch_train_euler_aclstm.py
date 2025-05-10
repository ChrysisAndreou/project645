import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse

# Hip_index = read_bvh.joint_index['hip'] # This is an index for positional data, not directly for raw BVH channels.
# For raw BVH data: Hip X pos is 0, Y pos is 1, Z pos is 2. Rotations start at index 3.

Seq_len=100 # Default, can be overridden by args
Hidden_size = 1024 # Default, can be overridden by args
# Joints_num =  57 # Specific to positional encoding's structure
Condition_num=5
Groundtruth_num=5
# In_frame_size = Joints_num*3 # This will be set by self.in_frame_size from args for Euler


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=171, hidden_size=1024, out_frame_size=171): # Default in_frame_size might change for Euler
        super(acLSTM, self).__init__()
        
        self.in_frame_size=in_frame_size
        self.hidden_size=hidden_size
        self.out_frame_size=out_frame_size
        
        ##lstm#########################################################
        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)#param+ID
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)
    
    
    #output: [batch*1024, batch*1024, batch*1024], [batch*1024, batch*1024, batch*1024]
    def init_hidden(self, batch):
        #c batch*(3*1024)
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h1= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        h2= torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size)) ).cuda())
        return  ([h0,h1,h2], [c0,c1,c2])
    
    #in_frame b*In_frame_size
    #vec_h [b*1024,b*1024,b*1024] vec_c [b*1024,b*1024,b*1024]
    #out_frame b*In_frame_size
    #vec_h_new [b*1024,b*1024,b*1024] vec_c_new [b*1024,b*1024,b*1024]
    def forward_lstm(self, in_frame, vec_h, vec_c):

        vec_h0,vec_c0=self.lstm1(in_frame, (vec_h[0],vec_c[0]))
        vec_h1,vec_c1=self.lstm2(vec_h[0], (vec_h[1],vec_c[1]))
        vec_h2,vec_c2=self.lstm3(vec_h[1], (vec_h[2],vec_c[2]))
     
        out_frame = self.decoder(vec_h2) #out b*out_frame_size (was 150 in comment, now general)
        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]
        
        
        return (out_frame,  vec_h_new, vec_c_new)
        
    #output numpy condition list in the form of [groundtruth_num of 1, condition_num of 0, groundtruth_num of 1, condition_num of 0,.....]
    def get_condition_lst(self,condition_num, groundtruth_num, seq_len ):
        gt_lst=np.ones((100,groundtruth_num)) # Original uses 100, check if related to max seq_len
        con_lst=np.zeros((100,condition_num))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]
        
    
    #in cuda tensor real_seq: b*seq_len*frame_size
    #out cuda tensor out_seq  b* (seq_len*frame_size)
    def forward(self, real_seq, condition_num=5, groundtruth_num=5):
        
        batch=real_seq.size()[0]
        seq_len=real_seq.size()[1]
        
        condition_lst=self.get_condition_lst(condition_num, groundtruth_num, seq_len)
        
        #initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)
        
        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda())

        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda())
        
        
        for i in range(seq_len):
            
            if(condition_lst[i]==1):##input groundtruth frame
                in_frame=real_seq[:,i]
            else:
                in_frame=out_frame
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
    
        return out_seq[:, 1: out_seq.size()[1]]
    
    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size) 
    def calculate_loss(self, out_seq, groundtruth_seq):
        # out_seq and groundtruth_seq are [batch_size, seq_len * self.out_frame_size]
        batch_size = out_seq.shape[0]
        
        num_elements = out_seq.shape[1]
        # Assuming self.out_frame_size is correctly set (e.g., 63 for standard BVH Euler)
        seq_len = num_elements // self.out_frame_size

        out_seq_reshaped = out_seq.reshape(batch_size, seq_len, self.out_frame_size)
        groundtruth_seq_reshaped = groundtruth_seq.reshape(batch_size, seq_len, self.out_frame_size)

        # Hip positions are the first 3 channels (Xpos, Ypos, Zpos)
        # In groundtruth_seq_reshaped: X and Z are differences, Y is absolute.
        hip_pos_pred = out_seq_reshaped[..., :3]
        hip_pos_gt = groundtruth_seq_reshaped[..., :3]
        
        # Rotations are from channel 3 onwards (Euler angles in degrees)
        rot_pred_deg = out_seq_reshaped[..., 3:]
        rot_gt_deg = groundtruth_seq_reshaped[..., 3:]

        loss_fn_mse = nn.MSELoss()
        loss_hip = loss_fn_mse(hip_pos_pred, hip_pos_gt)

        # Angle Distance Loss for rotations
        # AD = sum(1 - cos(pred_angle_rad - gt_angle_rad))
        # Convert degrees to radians for AD loss
        rot_pred_rad = rot_pred_deg * (np.pi / 180.0)
        rot_gt_rad = rot_gt_deg * (np.pi / 180.0)
        
        cos_error = torch.cos(rot_pred_rad - rot_gt_rad)
        # Mean over all angle components, sequence length, and batch.
        loss_rot_ad = torch.mean(1.0 - cos_error) 
        
        # Combine losses. Using simple sum. Weights might be needed (e.g. 1.0 for both).
        total_loss = loss_hip + loss_rot_ad 
        
        return total_loss


#numpy array real_seq_np: batch*seq_len*frame_size
def train_one_iteraton(real_seq_np, model, optimizer, iteration, save_dance_folder, print_loss=False, save_bvh_motion=True):
    # real_seq_np is [batch, seq_len_plus_2, frame_size]
    # frame_size is the number of channels in the Euler/BVH data (e.g., 63)

    # Calculate differences for hip X and Z positions. Hip Y and all rotations remain absolute.
    # dif_hip_x will be [batch, seq_len_plus_1]
    dif_hip_x = real_seq_np[:, 1:, 0] - real_seq_np[:, :-1, 0]
    dif_hip_z = real_seq_np[:, 1:, 2] - real_seq_np[:, :-1, 2]
    
    # real_seq_dif_hip_x_z_np will be [batch, seq_len_plus_1, frame_size]
    # It contains the sequence data that will be fed to the network (after slicing for input)
    # and used to form the ground truth (after slicing).
    real_seq_dif_hip_x_z_np = real_seq_np[:, :-1].copy() # Copy the sequence except the last frame for target
    real_seq_dif_hip_x_z_np[:, :, 0] = dif_hip_x         # Set Hip X to its difference
                                                        # Hip Y (index 1) remains absolute
    real_seq_dif_hip_x_z_np[:, :, 2] = dif_hip_z         # Set Hip Z to its difference
                                                        # All rotation channels (from index 3) remain absolute

    real_seq_cuda = torch.autograd.Variable(torch.FloatTensor(real_seq_dif_hip_x_z_np.tolist()).cuda())
 
    # seq_len for LSTM input (original Seq_len from args, e.g., 100)
    # real_seq_dif_hip_x_z_np has length seq_len_plus_1 (e.g. 101 if original Seq_len was 100)
    # So, input sequence is [:, 0:seq_len]
    # Target sequence is [:, 1:seq_len+1]
    
    # Current model.forward expects seq_len, so if args.seq_len is 100,
    # in_real_seq should be [batch, 100, frame_size]
    # predict_groundtruth_seq should be [batch, 100, frame_size]
    # The seq_len argument to train() is args.seq_len which is used for sampling, then incremented by 2.
    # Let's assume `model.forward` processes a sequence of `args.seq_len` frames.
    # So, if `real_seq_np` has `args.seq_len + 2` frames,
    # then `real_seq_dif_hip_x_z_np` has `args.seq_len + 1` frames.
    # `in_real_seq` should be the first `args.seq_len` frames of `real_seq_dif_hip_x_z_np`.
    # `predict_groundtruth_seq` should be the next `args.seq_len` frames (offset by 1).

    # `model.forward` takes `in_real_seq` of shape (batch, lstm_seq_len, frame_size)
    # lstm_seq_len is effectively args.seq_len (the original value before +2).
    # If `real_seq_cuda` has `args.seq_len + 1` frames (indices 0 to `args.seq_len`):
    in_lstm_seq_len = real_seq_cuda.size()[1] - 1 # This should be args.seq_len
    in_real_seq = real_seq_cuda[:, :in_lstm_seq_len] # First 'lstm_seq_len' frames
    
    # Ground truth for these 'lstm_seq_len' predictions
    # These are the 'lstm_seq_len' frames starting from the second frame of real_seq_dif_hip_x_z_np
    predict_groundtruth_seq_targets = real_seq_cuda[:, 1:in_lstm_seq_len+1] 
    
    # Flatten for the loss function as it expects (batch, seq_len * frame_size)
    predict_groundtruth_seq_flat = predict_groundtruth_seq_targets.reshape(real_seq_np.shape[0], -1)

    predict_seq = model.forward(in_real_seq, Condition_num, Groundtruth_num) # model.forward returns flat
    
    optimizer.zero_grad()

    loss = model.calculate_loss(predict_seq, predict_groundtruth_seq_flat)

    loss.backward()
    
    optimizer.step()
    
    if(print_loss==True):
        print ("###########"+"iter %07d"%iteration +"######################")
        print ("loss: "+str(loss.detach().cpu().numpy()))

    
    if(save_bvh_motion==True):
        # Save the first motion sequence in the batch.
        # predict_groundtruth_seq_targets is [batch, lstm_seq_len, frame_size]
        # predict_seq is [batch, lstm_seq_len * frame_size], need to reshape
        
        current_frame_size = model.out_frame_size # Should be same as in_frame_size for Euler
        
        gt_seq_to_save = np.array(predict_groundtruth_seq_targets[0].data.cpu().numpy()) # Already [lstm_seq_len, frame_size]
        
        # Reconstruct hip X and Z absolute positions for ground truth
        last_x_gt = 0.0 # Should initialize with actual starting pos if available, or assume 0 for delta sequence
        last_z_gt = 0.0
        # This assumes the first frame of gt_seq_to_save has hip_x, hip_z as deltas from a PREVIOUS base.
        # The `real_seq_np`'s first frame's hip_pos could be used as `last_x_gt`, `last_z_gt` initial values.
        # For simplicity matching original, let's assume relative reconstruction starts from 0,0.
        # Or, better, seed with the hip position from the *actual* first frame of the segment.
        # `real_seq_np[0,0,0]` is the hip X of the very first frame of the sampled (seq_len+2) segment.
        # `gt_seq_to_save` corresponds to frames 1 to `lstm_seq_len+1` of the original `real_seq_np`.
        # So, `last_x_gt` should be `real_seq_np[0, 0, 0]` and `last_z_gt` should be `real_seq_np[0, 0, 2]`.
        
        # To be precise: real_seq_dif_hip_x_z_np was formed from real_seq_np[:, :-1]
        # predict_groundtruth_seq_targets was real_seq_dif_hip_x_z_np[:, 1:]
        # This means predict_groundtruth_seq_targets[0, 0, :] corresponds to real_seq_np[0, 1, :] with X,Z as diffs from real_seq_np[0, 0, :]
        # So, the base for the first element of gt_seq_to_save's X,Z is real_seq_np[0,0,0] and real_seq_np[0,0,2]

        initial_hip_x = real_seq_np[0, 0, 0] 
        initial_hip_z = real_seq_np[0, 0, 2]
        
        # Make a copy for saving, as we modify in place
        gt_seq_bvh = gt_seq_to_save.copy()
        last_x_gt = initial_hip_x
        last_z_gt = initial_hip_z
        for frame_idx in range(gt_seq_bvh.shape[0]):
            # gt_seq_bvh[frame_idx, 0] is delta_x, gt_seq_bvh[frame_idx, 2] is delta_z
            current_delta_x = gt_seq_bvh[frame_idx, 0]
            current_delta_z = gt_seq_bvh[frame_idx, 2]
            gt_seq_bvh[frame_idx, 0] = last_x_gt + current_delta_x
            gt_seq_bvh[frame_idx, 2] = last_z_gt + current_delta_z
            last_x_gt = gt_seq_bvh[frame_idx, 0]
            last_z_gt = gt_seq_bvh[frame_idx, 2]
            # Hip Y (index 1) and rotations (index 3+) are already absolute.
        
        # Reshape predicted sequence and reconstruct hip positions
        out_seq_reshaped = np.array(predict_seq[0].data.cpu().numpy()).reshape(-1, current_frame_size)
        out_seq_bvh = out_seq_reshaped.copy()
        last_x_out = initial_hip_x # Start from the same initial hip pos as ground truth for fair comparison
        last_z_out = initial_hip_z
        for frame_idx in range(out_seq_bvh.shape[0]):
            current_delta_x = out_seq_bvh[frame_idx, 0]
            current_delta_z = out_seq_bvh[frame_idx, 2]
            out_seq_bvh[frame_idx, 0] = last_x_out + current_delta_x
            out_seq_bvh[frame_idx, 2] = last_z_out + current_delta_z
            last_x_out = out_seq_bvh[frame_idx, 0]
            last_z_out = out_seq_bvh[frame_idx, 2]

        read_bvh.write_traindata_to_bvh(save_dance_folder+"%07d"%iteration+"_gt.bvh", gt_seq_bvh)
        read_bvh.write_traindata_to_bvh(save_dance_folder+"%07d"%iteration+"_out.bvh", out_seq_bvh)

#input a list of dances [dance1, dance2, dance3]
#return a list of dance index, the occurence number of a dance's index is proportional to the length of the dance
def get_dance_len_lst(dances):
    len_lst=[]
    for dance in dances:
        #length=len(dance)/100 # Original heuristic
        length = max(1, dance.shape[0] // 100) # Ensure at least 1, use actual dance length
        len_lst=len_lst+[length]
    
    index_lst=[]
    current_idx=0 # Renamed from 'index' to avoid conflict with module
    for length in len_lst:
        for i in range(length): # Original used 'length' which was float, ensure int
            index_lst=index_lst+[current_idx]
        current_idx=current_idx+1
    return index_lst

#input dance_folder name
#output a list of dances.
def load_dances(dance_folder):
    dance_files=os.listdir(dance_folder)
    dances=[]
    print('Loading motion files...')
    for dance_file in dance_files:
        if not dance_file.endswith(".npy"): # Skip non-npy files
            continue
        # print ("load "+dance_file)
        dance_path = os.path.join(dance_folder, dance_file)
        try:
            dance=np.load(dance_path)
            dances=dances+[dance]
        except Exception as e:
            print(f"Could not load or process {dance_path}: {e}")
            
    print(len(dances), 'Motion files loaded.')
    if not dances:
        print(f"Warning: No dance files loaded from {dance_folder}. Training will likely fail.")
    return dances
    
# dances: [dance1, dance2, dance3,....]
def train(dances, frame_rate, batch_size_arg, seq_len_arg, read_weight_path, write_weight_folder,
          write_bvh_motion_folder, in_frame_arg, out_frame_arg, hidden_size_arg=1024, total_iter=100000): # Renamed args to avoid conflict
    
    # This is the effective sequence length used for LSTM steps, e.g., 100
    # The data sampling will take seq_len_arg + 2 frames
    lstm_seq_len = seq_len_arg 
    sample_seq_len = seq_len_arg + 2 # Total frames to sample from each dance for one training item (input + target)
    
    torch.cuda.set_device(0) # Make this configurable if multi-GPU

    model = acLSTM(in_frame_size=in_frame_arg, hidden_size=hidden_size_arg, out_frame_size=out_frame_arg)
    
    if(read_weight_path!=""):
        if os.path.exists(read_weight_path):
            model.load_state_dict(torch.load(read_weight_path))
            print(f"Loaded weights from {read_weight_path}")
        else:
            print(f"Weight file {read_weight_path} not found. Training from scratch.")

    model.cuda()
    #model=torch.nn.DataParallel(model, device_ids=[0,1]) # Example for multi-GPU

    current_lr=0.0001 # Learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)
    
    model.train()
    
    if not dances:
        print("No dances to train on. Exiting.")
        return

    dance_len_lst=get_dance_len_lst(dances)
    if not dance_len_lst:
        print("Dance length list is empty (no dances loaded or all dances too short?). Exiting.")
        return
    random_range=len(dance_len_lst)
    
    # speed ratio for frame rate adjustment, e.g. if dance is 60fps and network trained at 30fps, speed = 2
    speed=frame_rate/30 
    
    for iteration in range(total_iter):   
        #get a batch of dances
        dance_batch=[]
        for b_idx in range(batch_size_arg): # Renamed loop var
            # randomly pick up one dance. the longer the dance is the more likely the dance is picked up
            dance_id_idx = dance_len_lst[np.random.randint(0,random_range)] # Renamed loop var
            dance=dances[dance_id_idx].copy() # dance is [frames, channels]
            dance_len_frames = dance.shape[0]
            
            # Ensure enough frames for sampling: sample_seq_len frames at the given speed
            required_raw_frames = int(sample_seq_len * speed) + 1 # +1 for safety margin
            if dance_len_frames < required_raw_frames + 20: # +20 for start/end padding
                # print(f"Skipping dance {dance_id_idx} due to insufficient length ({dance_len_frames}) for {sample_seq_len} samples at speed {speed}")
                # Fallback: if a dance is too short, try another one, or pad (not implemented here)
                # For now, let's just resample if a short dance is picked. This could loop if all are too short.
                # A better way would be to filter short dances once at the start.
                if len(dances) > 1 : # Avoid infinite loop if only one short dance
                     b_idx -=1 # retry this batch item
                     continue
                else: # only one dance and it's too short. Pad or error.
                     print(f"Warning: Dance {dance_id_idx} is too short and it's the only/last one.")
                     # Simplistic padding: repeat last frame if necessary (not ideal for dynamics)
                     if dance_len_frames < required_raw_frames:
                         padding_frames = required_raw_frames - dance_len_frames
                         dance = np.concatenate([dance, np.tile(dance[-1:], (padding_frames, 1))], axis=0)
                         dance_len_frames = dance.shape[0]


            # start_id is in the original dance's frame indices
            # We need to pick `sample_seq_len` frames, considering `speed`.
            # Max start_id allows (sample_seq_len-1)*speed to be the last sampled index.
            max_start_id = dance_len_frames - (required_raw_frames) -10 # -10 for end padding
            start_id=random.randint(10, max(10, max_start_id) ) # Ensure start_id is at least 10
            
            sample_seq=[]
            for i in range(sample_seq_len): # e.g. 102 frames
                frame_to_sample = dance[int(i*speed+start_id)]
                sample_seq=sample_seq+[frame_to_sample]
            
            # augment the direction and position of the dance, helps the model to not overfeed
            # For Euler, augmentation might need to be careful about angle ranges if not handled by read_bvh.augment_train_data
            # The original augment_train_data works on positional. Assuming it's adapted or works for Euler as is.
            # Given it translates and rotates the whole character, it should be fine for Euler if it expects raw BVH like data.
            # However, read_bvh.augment_train_data converts to positional-like structure internally before augmenting.
            # This is problematic if `sample_seq` is already Euler.
            # For now, let's skip augmentation if using Euler directly, unless `augment_train_data` is verified/adapted.
            # For this version, I will comment out augmentation for Euler training.
            # T=[0.1*(random.random()-0.5),0.0, 0.1*(random.random()-0.5)]
            # R=[0,1,0,(random.random()-0.5)*np.pi*2]
            # sample_seq_augmented=read_bvh.augment_train_data(sample_seq, T, R) # This expects positional like data
            # dance_batch=dance_batch+[sample_seq_augmented]
            dance_batch=dance_batch+[sample_seq] # Use non-augmented for now for Euler
        
        if not dance_batch: # If all dances were too short and skipped
            print("Warning: dance_batch is empty. Skipping this iteration.")
            continue

        dance_batch_np=np.array(dance_batch) # Shape: [batch, sample_seq_len, frame_size]
       
        
        print_loss_flag=False # Renamed var
        save_bvh_motion_flag=False # Renamed var
        if(iteration % 20==0):
            print_loss_flag=True
        if(iteration % 1000==0 or iteration == total_iter -1): # Save on last iteration too
            save_bvh_motion_flag=True
            if not os.path.exists(write_weight_folder):
                os.makedirs(write_weight_folder)
            path = os.path.join(write_weight_folder, "%07d"%iteration +".weight")
            torch.save(model.state_dict(), path)
            
        train_one_iteraton(dance_batch_np, model, optimizer, iteration, write_bvh_motion_folder, print_loss_flag, save_bvh_motion_flag)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dances_folder', type=str, required=True, help='Path for the training data (npy files)')
    parser.add_argument('--write_weight_folder', type=str, required=True, help='Path to store checkpoints')
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True, help='Path to store test generated bvh during training')
    parser.add_argument('--read_weight_path', type=str, default="", help='Path to pre-trained checkpoint model path to continue training')
    
    # Important: dance_frame_rate is the original FPS of the BVH files from which npy were created.
    # The network internally might operate at a fixed 30 FPS by using 'speed' for sampling.
    parser.add_argument('--dance_frame_rate', type=int, default=60, help='Frame rate of the source BVH data')
    
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # For Euler (raw BVH), this is total channels per frame (e.g., 63 for standard.bvh)
    parser.add_argument('--in_frame', type=int, required=True, help='Input channels per frame for the model')
    parser.add_argument('--out_frame', type=int, required=True, help='Output channels per frame for the model (should match in_frame)')
    
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size of LSTM layers')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length for LSTM processing (number of frames)')
    parser.add_argument('--total_iterations', type=int, default=100000, help='Total training iterations')

    args = parser.parse_args()


    if not os.path.exists(args.write_weight_folder):
        os.makedirs(args.write_weight_folder)
    if not os.path.exists(args.write_bvh_motion_folder):
        os.makedirs(args.write_bvh_motion_folder)
    if not os.path.exists(args.dances_folder):
        print(f"ERROR: Dances folder {args.dances_folder} does not exist!")
        return

    dances_data = load_dances(args.dances_folder) # Renamed var

    if not dances_data:
        print(f"No data loaded from {args.dances_folder}. Please check the folder and its contents.")
        return

    train(dances_data, args.dance_frame_rate, args.batch_size, args.seq_len, 
          args.read_weight_path, args.write_weight_folder, args.write_bvh_motion_folder,
          args.in_frame, args.out_frame, args.hidden_size, total_iter=args.total_iterations)

if __name__ == '__main__':
    main()
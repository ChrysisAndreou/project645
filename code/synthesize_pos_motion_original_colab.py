import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
import read_bvh
import argparse

Hip_index = read_bvh.joint_index['hip']

Seq_len=100
Hidden_size = 1024
Joints_num =  57
Condition_num=5
Groundtruth_num=5
In_frame_size = Joints_num*3

class acLSTM(nn.Module):
    def __init__(self, in_frame_size=171, hidden_size=1024, out_frame_size=171):
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
     
        out_frame = self.decoder(vec_h2) #out b*150
        vec_h_new=[vec_h0, vec_h1, vec_h2]
        vec_c_new=[vec_c0, vec_c1, vec_c2]
        
        
        return (out_frame,  vec_h_new, vec_c_new)
        
    #output numpy condition list in the form of [groundtruth_num of 1, condition_num of 0, groundtruth_num of 1, condition_num of 0,.....]
    def get_condition_lst(self,condition_num, groundtruth_num, seq_len ):
        gt_lst=np.ones((100,groundtruth_num))
        con_lst=np.zeros((100,condition_num))
        lst=np.concatenate((gt_lst, con_lst),1).reshape(-1)
        return lst[0:seq_len]
        
    
    #in cuda tensor initial_seq: b*(initial_seq_len*frame_size)
    #out cuda tensor out_seq  b* ( (intial_seq_len + generate_frame_number) *frame_size)
    def forward(self, initial_seq, generate_frames_number):
        
        batch=initial_seq.size()[0]
        
        
        #initialize vec_h vec_m #set as 0
        (vec_h, vec_c) = self.init_hidden(batch)
        
        out_seq = torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,1))   ).cuda())

        out_frame=torch.autograd.Variable(torch.FloatTensor(  np.zeros((batch,self.out_frame_size))  ).cuda())
        
        
        for i in range(initial_seq.size()[1]):
            in_frame=initial_seq[:,i]
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
        
        for i in range(generate_frames_number):
            
            in_frame=out_frame
            
            (out_frame, vec_h,vec_c) = self.forward_lstm(in_frame, vec_h, vec_c)
    
            out_seq = torch.cat((out_seq, out_frame),1)
    
        return out_seq[:, 1: out_seq.size()[1]]
    
    #cuda tensor out_seq batch*(seq_len*frame_size)
    #cuda tensor groundtruth_seq batch*(seq_len*frame_size) 
    def calculate_loss(self, out_seq, groundtruth_seq):
        
        loss_function = nn.MSELoss()
        loss = loss_function(out_seq, groundtruth_seq)
        return loss


#numpy array inital_seq_np: batch*seq_len*frame_size
#return numpy b*generate_frames_number*frame_data
def generate_seq(initial_seq_np, generate_frames_number, model, save_dance_folder):

    #set hip_x and hip_z as the difference from the future frame to current frame
    dif = initial_seq_np[:, 1:initial_seq_np.shape[1]] - initial_seq_np[:, 0: initial_seq_np.shape[1]-1]
    initial_seq_dif_hip_x_z_np = initial_seq_np[:, 0:initial_seq_np.shape[1]-1].copy()
    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3]=dif[:,:,Hip_index*3]
    initial_seq_dif_hip_x_z_np[:,:,Hip_index*3+2]=dif[:,:,Hip_index*3+2]
    
    
    initial_seq  = torch.autograd.Variable(torch.FloatTensor(initial_seq_dif_hip_x_z_np.tolist()).cuda() )
 
    predict_seq = model.forward(initial_seq, generate_frames_number)
    
    batch=initial_seq_np.shape[0]
   
    for b in range(batch):
        
        out_seq=np.array(predict_seq[b].data.tolist()).reshape(-1,In_frame_size)
        last_x=0.0
        last_z=0.0
        for frame in range(out_seq.shape[0]):
            out_seq[frame,Hip_index*3]=out_seq[frame,Hip_index*3]+last_x
            last_x=out_seq[frame,Hip_index*3]
            
            out_seq[frame,Hip_index*3+2]=out_seq[frame,Hip_index*3+2]+last_z
            last_z=out_seq[frame,Hip_index*3+2]
            
        read_bvh.write_traindata_to_bvh(save_dance_folder+"out"+"%02d"%b+".bvh", out_seq)
    return np.array(predict_seq.data.tolist()).reshape(batch, -1, In_frame_size)



#input a list of dances [dance1, dance2, dance3]
#return a list of dance index, the occurence number of a dance's index is proportional to the length of the dance
def get_dance_len_lst(dances):
    len_lst=[]
    for dance in dances:
        length=len(dance)/100
        length=10
        if(length<1):
            length=1              
        len_lst=len_lst+[length]
    
    index_lst=[]
    index=0
    for length in len_lst:
        for i in range(length):
            index_lst=index_lst+[index]
        index=index+1
    return index_lst

#input dance_folder name
#output a list of dances.
def load_dances(dance_folder):
    dance_files=os.listdir(dance_folder)
    dances=[]
    for dance_file in dance_files:
        print ("load "+dance_file)
        dance=np.load(dance_folder+dance_file)
        print ("frame number: "+ str(dance.shape[0]))
        dances=dances+[dance]
    return dances
    
# dances: [dance1, dance2, dance3,....]
def test(dances, frame_rate, batch, initial_seq_len, generate_frames_number, read_weight_path,
         write_bvh_motion_folder, in_frame_size=171, hidden_size=1024, out_frame_size=171):
    
    torch.cuda.set_device(0)

    # Initialize and load pretrained weights
    model = acLSTM(in_frame_size, hidden_size, out_frame_size)
    model.load_state_dict(torch.load(read_weight_path))
    # Move model to GPU
    model.cuda()

    dance_len_lst=get_dance_len_lst(dances)
    random_range=len(dance_len_lst)
    
    speed=frame_rate/30 # we train the network with frame rate of 30
    
    dance_batch=[]
    # Get 15 step sequence initial motion
    for b in range(batch):
        #randomly pick up one dance. the longer the dance is the more likely the dance is picked up
        dance_id = dance_len_lst[np.random.randint(0,random_range)]
        dance=dances[dance_id].copy()
        dance_len = dance.shape[0]
            
        start_id=random.randint(10, int(dance_len-initial_seq_len*speed-10))#the first and last several frames are sometimes noisy. 
        sample_seq=[]
        for i in range(initial_seq_len):
            sample_seq=sample_seq+[dance[int(i*speed+start_id)]]
        
        dance_batch=dance_batch+[sample_seq]
            
    dance_batch_np=np.array(dance_batch)

    # Generate the next steps
    generate_seq(dance_batch_np, generate_frames_number, model, write_bvh_motion_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthesize motion using a trained acLSTM model (Original Script Version).")
    parser.add_argument('--read_weight_path', type=str, required=True, help='Path to the trained model weights (.weight file)')
    parser.add_argument('--dances_folder', type=str, required=True, help='Path to the folder containing seed motion .npy files')
    parser.add_argument('--write_bvh_motion_folder', type=str, required=True, help='Path to the folder to save generated BVH files')
    
    parser.add_argument('--dance_frame_rate', type=int, default=60, help='Frame rate of the dance data (default: 60)')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of sequences to generate in a batch (default: 5)')
    parser.add_argument('--initial_seq_len', type=int, default=15, help='Length of the initial seed motion sequence (frames) (default: 15)')
    parser.add_argument('--generate_frames_number', type=int, default=400, help='Number of frames to synthesize (default: 400)')
    
    # Default values from the script's global constants or test() function defaults
    # These global constants (In_frame_size, Hidden_size) are defined at the top of the script
    parser.add_argument('--in_frame_size', type=int, default=In_frame_size, help=f'Input frame size for the model (default: {In_frame_size})')
    parser.add_argument('--hidden_size', type=int, default=Hidden_size, help=f'Hidden size of the LSTM layers (default: {Hidden_size})')
    parser.add_argument('--out_frame_size', type=int, default=In_frame_size, help=f'Output frame size for the model (default: {In_frame_size})')

    args = parser.parse_args()

    if not os.path.exists(args.write_bvh_motion_folder):
        os.makedirs(args.write_bvh_motion_folder)

    dances_list = load_dances(args.dances_folder)
    if not dances_list:
        print(f"Error: No dance files found in {args.dances_folder} or failed to load. Exiting.")
        exit()

    print(f"Starting synthesis with original script logic...")
    print(f"  Config:")
    print(f"    Read weight path: {args.read_weight_path}")
    print(f"    Dances folder: {args.dances_folder}")
    print(f"    Write BVH motion folder: {args.write_bvh_motion_folder}")
    print(f"    Dance frame rate: {args.dance_frame_rate}")
    print(f"    Batch size: {args.batch_size}")
    print(f"    Initial sequence length: {args.initial_seq_len}")
    print(f"    Generated frames number: {args.generate_frames_number}")
    print(f"    In frame size: {args.in_frame_size}")
    print(f"    Hidden size: {args.hidden_size}")
    print(f"    Out frame size: {args.out_frame_size}")
    
    # The first argument to test() is the list of loaded dances.
    # The parameter name 'dances' in test() definition refers to this list.
    test(
        dances_list,
        args.dance_frame_rate,
        args.batch_size, # Corresponds to 'batch' in test() signature
        args.initial_seq_len,
        args.generate_frames_number,
        args.read_weight_path,
        args.write_bvh_motion_folder,
        in_frame_size=args.in_frame_size,
        hidden_size=args.hidden_size,
        out_frame_size=args.out_frame_size
    )
    
    print(f"Original script synthesis complete. BVH files saved to: {args.write_bvh_motion_folder}")
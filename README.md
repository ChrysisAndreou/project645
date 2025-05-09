# Project - Character Motion Prediction
This folder contains an implementation of acRNN written in Pytorch.

See the following link for more background:

[Auto-Conditioned Recurrent Networks for Extended Complex Human Motion Synthesis](https://arxiv.org/abs/1707.05363)

### Prequisite

If you running the code locally you can create the conda enviroment. Run the following commands:
```
conda env create -f mai645.yml
source activate mai645
```

### Data Preparation

The bvh motion files including "salsa", "martial" and "indian" are included in the "train_data_bvh" folder.

Then to transform the bvh files into training data, go to the folder "code" and run:
```
python code/generate_training_pos_data.py
```

You will need to change the directory of the source motion folder and the target motion folder. If you don't change anything, this code will create a directory "./train_data_pos/martial" and generate the training data for martial dances in this folder for positional encoding.

### Training

After generating the training data, you can start to train the network by running:
```
python code/pytorch_train_pos_aclstm.py --dances_folder <DIR> --write_weight_folder <DIR> --write_bvh_motion_folder <DIR> --in_frame 171 --out_frame 171 --batch_size <INT>
```
You need to define some directories on the input arguments, including "dances_folder" which is the location of the training data, "write_weight_folder" which is 
the location to save the weights of the network during training, "write_bvh_motion_folder" which is the location to save the temporate output of the network and the groundtruth motion sequences in the form of bvh, and "read_weight_path" which is the path of the network weights if you want to train the network from some pretrained weights other than from begining in which case it is set as "". 

### Testing

When the training is done, you can synthesize motions by running:
```
python code/synthesize_pos_motion.py
```

For rendering the bvh motion, you can use softwares like MotionBuilder, Maya, 3D max or most easily, use an online BVH renderer for example:
http://lo-th.github.io/olympe/BVH_player.html 

Good Luck!

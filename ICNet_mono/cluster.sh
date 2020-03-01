#!/bin/sh
 
export OMP_NUM_THREADS=1 # needed to be able to run batchgenerators with new numpy versions. Use this!
 
module load python/3.6.1 # load python module. If you don't do this then you have no python
 
#source /home/h295d/virtualenv/tensorflow/bin/activate # activate my python virtual environment

source /home/h295d/.local/bin/virtualenvwrapper.sh
workon tf
 
CUDA=10.0
module load cudnn/7.6.1.34/cuda10.0
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH # check
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH # check
export PATH=/usr/local/cuda-${CUDA}/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA}/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-${CUDA}
 
 
 
DATASET_LOCATION=/datasets/datasets_h295d/Liver_data/crop_multi_data/
#DATASET_LOCATION=/datasets/datasets_h295d/Liver_data/crop/
#DATASET_LOCATION=/datasets/datasets_h295d/Liver_data/1mm_crop/
#DATASET_LOCATION=/datasets/datasets_h295d/liver_crop/affine_data/
#DATASET_LOCATION=/datasets/datasets_h295d/liver_with_bounding_box/affine_whole_16/
 
tmp_dir=/ssd/h295d/${LSB_JOBID} # this directors is automatically created by LSF and does not need to be created here
tmp_dir_data=${tmp_dir}/data # this is where I will store my dataset
mkdir $tmp_dir_data
tmp_dir_cache=${tmp_dir}/cache # needed for cuda cache
mkdir $tmp_dir_cache
 
CUDA_CACHE_PATH=$tmp_dir_cache
export CUDA_CACHE_PATH

echo $CUDA_CACHE_PATH
 
rsync -rtvu ${DATASET_LOCATION}/ ${tmp_dir_data} # this copies the data to the SSD
 
 
# I run this runner to run a number of experiments so whatever I want to run 
# I pass along as arguments when launching the job



#python3 /home/h295d/sourcecode/src/train_miccai2018.py --data_dir ${tmp_dir_data}
python3 /home/h295d/ICNet_Mind/Code/Train.py --datapath ${tmp_dir_data}

#python3 /datasets/datasets_h295d/ICNet/Code/Train.py --datapath ${tmp_dir_data}
#python3 /home/h295d/ICNet_Mind/Code/Train.py --datapath ${tmp_dir_data}

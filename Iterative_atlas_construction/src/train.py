import os
import glob
import sys
import random
from argparse import ArgumentParser
# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model 
# project imports
import datagenerators
import networks
import losses
sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen
import nibabel as nib
import matplotlib.pyplot as plt
import skimage
import tensorboard
import nibabel as nib
import itertools

vol_names = glob.glob(os.path.join('/data/My_Thesis/Code/voxelmorph/Freesurfer_affine/', '*.nii.gz'))

import networks

vol_size=(160,192,224)
nf_enc = [16,32,32,32]
nf_dec = [32,32,32,32,16,3]
net = networks.miccai2018_net(vol_size, nf_enc, nf_dec, use_miccai_int=False, indexing='ij')  
net.load_weights('/data/My_Thesis/Code/voxelmorph/Models/Freesurfer_model/1500.h5')
fixed=nib.load('../data/atlas_norm.nii.gz').get_data()

def train(model,
          data_dir,
          atlas_file,
          model_dir,
          gpu_id,
          lr,
          nb_epochs,
          prior_lambda,
          image_sigma,
          steps_per_epoch,
          batch_size,
          bidir,
          initial_epoch=0):
    
    # load atlas from provided files. The atlas we used is 160x192x224.
    atlas_vol = atlas_file[np.newaxis, ..., np.newaxis]
    vol_size = atlas_vol.shape[1:-1] 
    train_vol_names = glob.glob(os.path.join(data_dir, '*.nii.gz'))
    

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # load initial weights
    # save first iteration
    model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))
    
    
    flow_vol_shape = model.outputs[-1].shape[1:-1]
    loss_class = losses.Miccai2018(image_sigma, prior_lambda, flow_vol_shape=flow_vol_shape)
    if bidir:
        model_losses = [loss_class.recon_loss, loss_class.recon_loss, loss_class.kl_loss]
        loss_weights = [0.5, 0.5, 1]
    else:
        model_losses = [loss_class.recon_loss, loss_class.kl_loss]
        loss_weights = [1, 1]

    train_example_gen = datagenerators.example_gen1(train_vol_names, batch_size=batch_size)
    atlas_vol_bs = np.repeat(atlas_vol, batch_size, axis=0)
    miccai2018_gen = datagenerators.miccai2018_gen(train_example_gen,
                                                   atlas_vol_bs,
                                                   batch_size=batch_size,
                                                   bidir=bidir)

    # prepare callbacks
    save_file_name = os.path.join(model_dir,'{epoch:02d}.h5')
    save_callback = ModelCheckpoint(save_file_name)
    mg_model = model

    mg_model.compile(optimizer=Adam(lr=lr), loss=model_losses, loss_weights=loss_weights)
    mg_model.fit_generator(miccai2018_gen, 
                           initial_epoch=initial_epoch,
                           epochs=nb_epochs,
                           callbacks=[save_callback],
                           steps_per_epoch=steps_per_epoch,
                           verbose=1)
    return mg_model


#------------------------------- Iterative algotithum--------------------------------

error=1
sample=[]
j=0
while error>1e-7:
    wraped=[]
    fixed=fixed[np.newaxis, ..., np.newaxis]
    for i in range(len(vol_names)):
        moving=nib.load(vol_names[i]).get_data()[np.newaxis, ..., np.newaxis]/255
        [moved, warp] = net.predict([moving,fixed])
        wraped.append(moved[0,...,0])
    a=np.average(wraped,0)
    error=np.mean(np.square(fixed[0,...,0]-a))
    sample.append(a)
    print(error)
    fixed=a
    print(np.shape(fixed))
    net=train(net,'/data/My_Thesis/Code/voxelmorph/Freesurfer_affine/',fixed,'/data/My_Thesis/Code/voxelmorph/Models/iteration_'+str(j),'0',1e-4,50,25,0.01,150,1,0,0)
    j=j+1



import numpy as np
import keras
import glob
import os, sys
import matplotlib.pyplot as plt
import itertools
import skimage
import nibabel as nib
import SimpleITK as sitk

	# conver to nii after freesurfer.........................................................
def convert_nii(Dir):

	for f in os.listdir(Dir):
		if(f!=[]):
			os.system('mri_convert -i /data/matheias/datasets/datasets_h295d/freesurfer_seg/seg_freesurfer/'+str(f)+'/mri/T1.mgz -o /data/datasets/datasets_h295d/freesurfer_seg/vol_and_seg/'+str(f)+'/T1w.nii.gz')
			os.system('mri_convert -i /data/matheias/datasets/datasets_h295d/freesurfer_seg/seg_freesurfer/'+str(f)+'/mri/aseg.auto.mgz -o /data/matheias/datasets_h295d/freesurfer_seg/vol_and_seg/'+str(f)+'/seg.nii.gz')
			
			
convert_nii('/data/matheias/datasets/datasets_h295d/freesurfer_seg/seg_freesurfer/')



	# conver to affine aligned.........................................................

def affine_freesurfer(move,seg):

	where_to_store='/data/matheias/datasets/datasets_h295d/freesurfer_seg/affine'
	if not os.path.exists(os.path.join(where_to_store, move.split("/")[7])):
		os.makedirs(os.path.join(where_to_store, move.split("/")[7]))
	out_img=os.path.join(where_to_store, move.split("/")[7],move.split("/")[8])
	out_seg=os.path.join(where_to_store, move.split("/")[7],seg.split("/")[8])
	out=os.path.join(where_to_store, move.split("/")[7],'T1')
	out_seg=os.path.join(where_to_store, move.split("/")[7])

	os.system('bash antsRegistrationSyN.sh -d 3 -m '+move+' -f /data/My_Thesis/Code/new_code/voxelmorph/data/atlas.nii.gz -o '+out+' -t a -n 8')


	os.system('/home/hfahad/ANTs/bin/bin/antsApplyTransforms -d 3 -i '+seg+' -r /data/My_Thesis/Code/new_code/voxelmorph/data/atlas_norm.nii.gz --interpolation NearestNeighbor -o '+out_seg+'/seg.nii.gz -t '+out_seg+'/T10GenericAffine.mat')


def load_volume(Dir):
	f_3 = []
	f_6 = []
	mask= []
	seg = []
	fixed_seg_PZ = []
	moving_seg_PZ = []
	extension='.nii.gz'
	for f in os.listdir(Dir):
		if(f!=[]):

			moving = os.path.join(Dir, f,'T1w'+extension)
			seg = os.path.join(Dir, f, 'seg'+extension)

			affine_freesurfer(moving,seg)
load_volume('/data/matheias/datasets/datasets_h295d/freesurfer_seg/same_orig')








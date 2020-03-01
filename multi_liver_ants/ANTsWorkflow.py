import numpy as np
import sys
#import SimpleITK as sitk
import subprocess
import itertools
import os
import glob

def load_names(DIR):
    DIR = os.path.join(DIR, '*')
    b=glob.glob(DIR)
    M,M_seg=[],[]
    for i in b:
        if i.split('/')[-1][:12]:
            paths=glob.glob(i+'/*')
            c,d=[],[]
            if len(paths)==3:            
                for j in paths:
                    a=j.split('/')
                    if a[-1].startswith('00_seg'):
                        M_seg.append(j)
                    elif a[-1].split('_')[3]=='8':
                        M.append(j)
    F='/data/Liver_part/Multi_modality_data/700005_HELENA_20150527/700005_HELENA_20150527_11_3D_GRE_TRA_HELENA_W.nrrd'
    F_seg='/data/Liver_part/Multi_modality_data/700005_HELENA_20150527/00_seg, 30.08.2017, 700005_HELENA_20150527_11_3D_GRE_TRA_HELENA_W.nrrd'
    return F, M, F_seg, M_seg

def ANTs_on_data(Dir):

    F, M, F_seg, M_seg= load_names(Dir)
    out_DIR='/data/fahad_thesis/Multi-modal/Result'
    for i in range(len(M)):

        f_seg, m_seg=[],[]
        out_f_seg, out_m_seg=[], []
        '''Generating the output filenames'''
        if not os.path.exists(os.path.join(out_DIR, M[i].split("/")[-2])):
            os.makedirs(os.path.join(out_DIR, M[i].split("/")[-2]))

        out_M_img = os.path.join(out_DIR,M[i].split('/')[-2], M[i].split('/')[-1])
        field = os.path.join(out_DIR,M[i].split('/')[-2], 'field')

        command = ["/home/hfahad/ANTs/bin/bin/antsRegistration", '--dimensionality', '3',
                   '--write-composite-transform', '1','--output', '['+field+','+out_M_img+']', '--float','0',\
                 '--interpolation', 'Linear', '--winsorize-image-intensities', '[0.005,0.995]',\
                  '--use-histogram-matching', '0', '--verbose', '1', '--initial-moving-transform','['+F+','+M[i]+',1]',\
		'--transform', 'SyN[0.25]', '--metric', 'MI['+F+','+M[i]+',1,32,Regular,0.2]',\
		'--convergence',\
                '[500x500x500x500,1e-9,20]', '--shrink-factors', '8x4x2x1', '--smoothing-sigmas', '3x2x1x0vox',\
                ]
        cmd = subprocess.Popen(command)
        cmd.communicate()

        out_m_seg = os.path.join(out_DIR,M_seg[i].split('/')[-2],M_seg[i].split('/')[-1])

        command = ["/home/hfahad/ANTs/bin/bin/antsApplyTransforms", '-d', '3', '-i', M_seg[i], '-r', F,
           '--interpolation', 'NearestNeighbor', '-o', out_m_seg, '-t', field+'Composite.h5']
        cmd = subprocess.Popen(command)
        cmd.communicate()

print('Program is start')
ANTs_on_data('/data/Liver_part/Multi_modality_data/crop_multi_data')



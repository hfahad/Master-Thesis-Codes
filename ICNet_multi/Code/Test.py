import os
from argparse import ArgumentParser
import numpy as np
import torch
from torch.autograd import Variable
from Models import ModelFlow_stride,SpatialTransform
from Functions import generate_grid,load_5D,save_img,save_flow,load_5D_seg
import timeit
import glob

parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='/data/matheias/datasets/datasets_h295d/model_icnet_liver/15000.pth',
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='/data/newsyapa/Liver_results/mono_icnet_new',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8, 
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='/data/My_Thesis/Code/new_code/voxelmorph/data/atlas_norm.nii.gz', 
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='/data/My_Thesis/Code/voxelmorph/Freesurfer_affine/T1_70Warped.nii.gz', 
                    help="moving image")
opt = parser.parse_args()

savepath = opt.savepath
if not os.path.isdir(savepath):
    os.mkdir(savepath)
    
    
    

def test(f,m,mseg):
    model =ModelFlow_stride(2,3,8).cuda()
    transform = SpatialTransform().cuda()
    
    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()
    transform.eval()

    grid = generate_grid(imgshape)
    grid = Variable(torch.from_numpy(np.reshape(grid, (1,) + grid.shape))).cuda().float()

    start = timeit.default_timer()


    A = Variable(torch.from_numpy( load_5D(f))).cuda().float()
    B = Variable(torch.from_numpy( load_5D(m))).cuda().float()
    start2 = timeit.default_timer()
    print('Time for loading data: ', start2 - start) 
    pred = model(A,B)
    F_AB = pred.permute(0,2,3,4,1).data.cpu().numpy()[0, :, :, :, :]  
    F_AB = F_AB.astype(np.float32)*range_flow
    warped_A = transform(A,pred.permute(0,2,3,4,1)*range_flow,grid,'bilinear').data.cpu().numpy()[0, 0, :, :, :]
    start3 = timeit.default_timer()    
    print('Time for registration: ', start3 - start2)      
      
    warped_F_BA = transform(-pred,pred.permute(0,2,3,4,1)*range_flow,grid,'bilinear').permute(0,2,3,4,1).data.cpu().numpy()[0, :, :, :, :] 
    warped_F_BA = warped_F_BA.astype(np.float32)*range_flow 
 
    start4 = timeit.default_timer()    
    print('Time for generating inverse flow: ', start4 - start3)         
    
    #save_flow(F_AB,savepath+'/flow_A_B.nii.gz')      
    #save_flow(warped_F_BA,savepath+'/inverse_flow_B_A.nii.gz')   
    #save_img(warped_A,savepath+'/warped_A.nii.gz')     

    start5 = timeit.default_timer()    
    print('Time for saving results: ', start5 - start4)         
    del pred
    
    pred = model(B,A)
    F_BA = pred.permute(0,2,3,4,1).data.cpu().numpy()[0, :, :, :, :] 
    F_BA = F_BA.astype(np.float32)*range_flow     
    warped_B = transform(B,pred.permute(0,2,3,4,1)*range_flow,grid,'bilinear').data.cpu().numpy()[0, 0, :, :, :]
    warped_F_AB = transform(-pred,pred.permute(0,2,3,4,1)*range_flow,grid,'bilinear').permute(0,2,3,4,1).data.cpu().numpy()[0, :, :, :, :]    
    warped_F_AB = warped_F_AB.astype(np.float32)*range_flow
    #save_flow(F_BA,savepath+'/flow_B_A.nii.gz')      
    #save_flow(warped_F_AB,savepath+'/inverse_flow_A_B.nii.gz')   
    if not os.path.exists(os.path.join(savepath, m.split("/")[-2])):
        os.makedirs(os.path.join(savepath, m.split("/")[-2]))
    out_img=os.path.join(savepath, m.split("/")[-2],m.split('/')[-1])
    out_seg=os.path.join(savepath, m.split("/")[-2],mseg.split('/')[-1])
    B_seg = Variable(torch.from_numpy(load_5D_seg(mseg))).cuda().float()
    warped_B_seg = transform(B_seg,pred.permute(0,2,3,4,1)*range_flow,grid,'nearest').data.cpu().numpy()[0, 0, :, :, :]
    save_img(warped_B,out_img)
    #save_img(A,'/data/newsyapa/ICNet_results_pretrained/fix.nii.gz')
    save_img(warped_B_seg,out_seg)           



#------------------------------- For Liver data loader------------------------------
imgshape = (80, 208, 256)
range_flow = 7

def load_names(DIR):
    DIR = os.path.join(DIR, '*')
    b=glob.glob(DIR)
    M,M_seg=[],[]
    for i in b:
        if i.split('/')[-1][:12]:# in ['Prisma_00142', 'Prisma_00145','Prisma_00149','Prisma_00096','Prisma_00155', 'Prisma_00181','Prisma_00186','Prisma_00192','Prisma_00197']:
        #if i.split('/')[-1][:12] == 'Prisma_00082':
            paths=glob.glob(i+'/*')
            c,d=[],[]
            if len(paths)==3:            
                for j in paths:
                    a=j.split('/')
                    if a[-1].startswith('00_seg'):
                        M_seg.append(j)
                    elif a[-1].split('_')[3]=='11':
                        M.append(j)
    F='/data/Liver_part/Multi_modality_data/700005_HELENA_20150527/700005_HELENA_20150527_11_3D_GRE_TRA_HELENA_W.nrrd'
    F_seg='/data/Liver_part/Multi_modality_data/700005_HELENA_20150527/00_seg, 30.08.2017, 700005_HELENA_20150527_11_3D_GRE_TRA_HELENA_W.nrrd'
    return F, M, F_seg, M_seg

f,m,fseg,mseg=load_names('/data/Liver_part/Multi_modality_data/crop_multi_data')



##----------------------------------  For brain data loader------------------------------------------
'''
import itertools
def load_brain():
    Dir='/data/newsyapa/affine'
    move=[]
    seg=[]
    extension='.nii.gz'
    print(os.listdir(Dir))
    for f in os.listdir(Dir):
        if(f!=[]):
            move.append(os.path.join(Dir,f,'T1_hist'+extension))
            seg.append(os.path.join(Dir, f, 'seg'+extension))
#train_names= list(itertools.chain(*moving))
#seg_name=list(itertools.chain(*seg))
    fix='/data/newsyapa/affine/102109/T1_hist.nii.gz'
    return move,seg,fix
'''
#m,mseg,f=load_brain()
#imgshape = (160, 192,224)
#range_flow = 7

for i in range (len(m)):
	test(f,m[i],mseg[i])
    
    

    
    
    
    

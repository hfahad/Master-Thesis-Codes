



#--------------------- Same origan as a fixed volume-----------------
def affine_origin(fixed,fixed_seg,moving,moving1, f_seg):
    where_to_store='/data/Liver_part/liver_crop/same_origin/'
    out_img_name=os.path.join(where_to_store, moving.split("/")[5],moving.split("/")[6])
    out_img1_name=os.path.join(where_to_store, moving1.split("/")[5],moving1.split("/")[6])

    if not os.path.exists(os.path.join(where_to_store, moving.split("/")[5])):
        os.makedirs(os.path.join(where_to_store, moving.split("/")[5]))
    
    
    fix=sitk.ReadImage(fixed)
    fix_seg=sitk.ReadImage(fixed_seg)

    move=sitk.ReadImage(moving)
    move.SetOrigin(fix.GetOrigin())
    sitk.WriteImage(move,out_img_name)
    move1=sitk.ReadImage(moving1)
    move1.SetOrigin(fix_seg.GetOrigin())
    sitk.WriteImage(move1,out_img1_name)
    
    if f_seg is not None:
        out_seg_name=os.path.join(where_to_store, moving.split("/")[5],f_seg.split("/")[6])
        seg=sitk.ReadImage(f_seg)
        seg.SetOrigin(fix.GetOrigin())
        sitk.WriteImage(seg,out_seg_name)

def load_volume_origin(Dir):
    f_3 = []
    f_6 = []
    mask= []
    seg = []
    fixed_seg_PZ = []
    moving_seg_PZ = []
    extension='.nrrd'
    for f in os.listdir(Dir):
        if(f!=[]):
            fixed = '/data/Liver_part/Code/raw/700005_HELENA_20150527/700005_HELENA_20150527_11_3D_GRE_TRA_HELENA_W.nrrd'
            fixed_seg = '/data/Liver_part/Code/raw/700005_HELENA_20150527/700005_HELENA_20150527_34_FAT_QUANT_fl3d-vibe-Dixon_bh_Output_W-Liver.nrrd'
            #f_6 = os.path.join(Dir, f, f+'_34_FAT_QUANT_fl3d-vibe-Dixon_bh_Output_W'+extension)
            moving = os.path.join(Dir, f, f+'_11_3D_GRE_TRA_HELENA_W'+extension)
            moving1 = os.path.join(Dir, f, f+'_34_FAT_QUANT_fl3d-vibe-Dixon_bh_Output_W-Liver'+extension)
            seg = glob.glob(os.path.join(Dir, f, '00_seg*'+extension))

            if(seg!=[]): 
                affine_origin(fixed,fixed_seg, moving,moving1, seg[0])
            else:
                affine_origin(fixed,fixed_seg, moving,moving1, None)
load_volume_origin('/data/Liver_part/Multi_modality_data/raw')


#----------------------------affine registration and crop by factor 16----------
def affine_transformation_crop(fixed, moving,moving1, f_seg):
    where_to_store='/data/Liver_part/Multi_modality_data/crop_multi_data/'
    out_img_name=os.path.join(where_to_store, moving.split("/")[5],moving.split("/")[6])
    out_img1_name=os.path.join(where_to_store, moving1.split("/")[5],moving1.split("/")[6])
    
    if not os.path.exists(os.path.join(where_to_store, moving.split("/")[5])):
        os.makedirs(os.path.join(where_to_store, moving.split("/")[5]))
    #copyfile(fixed,os.path.join(where_to_store, fixed.split("/")[6],'T2_resampled.nrrd'))       
    #copyfile(f_seg_PRO,os.path.join(where_to_store, fixed.split("/")[6],'seg_T2_PRO.nrrd'))
    #copyfile(f_seg_PZ,os.path.join(where_to_store, fixed.split("/")[6],'seg_T2_PZ.nrrd'))  
                              
    elastixImageFilter = sitk.ElastixImageFilter()
    parameterMapVector = sitk.VectorOfParameterMap()
    param_rigid = sitk.GetDefaultParameterMap("rigid")
    param_affine = sitk.GetDefaultParameterMap("affine")
    #param_nonrigid = sitk.GetDefaultParameterMap("bspline")
    parameterMapVector.append(param_rigid)
    parameterMapVector.append(param_affine)
    #parameterMapVector.append(param_nonrigid)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    
    # extra
    a=sitk.GetArrayFromImage(sitk.ReadImage(fixed))[4:-12,39:-13,39:-25]
    fix=sitk.GetImageFromArray(a)
    fix.SetSpacing((1.40625, 1.40625, 3.0))
                               
    b=sitk.GetArrayFromImage(sitk.ReadImage(moving))[4:-12,39:-13,39:-25]
    move=sitk.GetImageFromArray(b)
    move.SetSpacing((1.40625, 1.40625, 3.0))
    c=sitk.GetArrayFromImage(sitk.ReadImage(moving1))[4:-12,39:-13,39:-25]
    move1=sitk.GetImageFromArray(c)
    move1.SetSpacing((1.40625, 1.40625, 3.0))
    
    elastixImageFilter.SetFixedImage(fix)
    elastixImageFilter.SetMovingImage(move)
    elastixImageFilter.Execute()
    reg_result = elastixImageFilter.GetResultImage()
    sitk.WriteImage(elastixImageFilter.GetResultImage(),out_img_name)
                             
                             
    transform_param_map = elastixImageFilter.GetTransformParameterMap()
    transform_param_map[0]['FinalBSplineInterpolationOrder'] = ['3']
    transform_param_map[1]['FinalBSplineInterpolationOrder'] = ['3']
    transform_param_map[0]['ResultImageFormat'] = ['nrrd']
    transform_param_map[1]['ResultImageFormat'] = ['nrrd']
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transform_param_map)
    transformixImageFilter.SetMovingImage(move1)
    transformixImageFilter.Execute()
    reg_struct_result = transformixImageFilter.GetResultImage()   
    sitk.WriteImage(reg_struct_result,out_img1_name)
    if f_seg is not None:                     
    #.............................................................
        out_seg_PRO=os.path.join(where_to_store, moving.split("/")[5],f_seg.split("/")[6])
        transform_param_map = elastixImageFilter.GetTransformParameterMap()
        transform_param_map[0]['FinalBSplineInterpolationOrder'] = ['0']
        transform_param_map[1]['FinalBSplineInterpolationOrder'] = ['0']
        transform_param_map[0]['ResultImageFormat'] = ['nrrd']
        transform_param_map[1]['ResultImageFormat'] = ['nrrd']
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(transform_param_map)
        
        d=sitk.GetArrayFromImage(sitk.ReadImage(f_seg))[4:-12,39:-13,39:-25]
        f_seg1=sitk.GetImageFromArray(d)
        f_seg1.SetSpacing((1.40625, 1.40625, 3.0))
    
        transformixImageFilter.SetMovingImage(f_seg1)
        transformixImageFilter.Execute()
        reg_struct_result = transformixImageFilter.GetResultImage()
        im = sitk.Cast(reg_struct_result, sitk.sitkUInt8)
        sitk.WriteImage(im,out_seg_PRO)
        #.............................................................

def load_volume_crop(Dir):
    f_3 = []
    f_6 = []
    mask= []
    seg = []
    fixed_seg_PZ = []
    moving_seg_PZ = []
    extension='.nrrd'
    for f in os.listdir(Dir):
        if(f!=[]):
            print(f)
            fixed = '/data/Liver_part/Multi_modality_data/affine_whole_image/700005_HELENA_20150527/700005_HELENA_20150527_11_3D_GRE_TRA_HELENA_W.nrrd'
            moving = os.path.join(Dir, f, f+'_11_3D_GRE_TRA_HELENA_W'+extension)
            moving1 = os.path.join(Dir, f, f+'_8_3D_GRE_TRA_HELENA_opp'+extension)
            seg = glob.glob(os.path.join(Dir, f, '00_seg*'+extension))

            if(seg!=[]): 
                affine_transformation_crop(fixed, moving,moving1, seg[0])
            else:
                affine_transformation_crop(fixed, moving,moving1, None)
load_volume_crop('/data/Liver_part/Multi_modality_data/raw/')






#---------------------------------------------------------warped grid Visualization---------------------------------------------------------
def bw_grid(vol_shape, spacing, thickness=1):
    """
    draw a black and white ND grid.
    Parameters
    ----------
        vol_shape: expected volume size
        spacing: scalar or list the same size as vol_shape
    Returns
    -------
        grid_vol: a volume the size of vol_shape with white lines on black background
    """

    # check inputs
    if not isinstance(spacing, (list, tuple)):
        spacing = [spacing] * len(vol_shape)
    assert len(vol_shape) == len(spacing)

    # go through axes
    grid_image = np.zeros(vol_shape)
    for d, v in enumerate(vol_shape):
        rng = [np.arange(0, f) for f in vol_shape]
        for t in range(thickness):
            rng[d] = np.append(np.arange(0+t, v, spacing[d]), -1)
            grid_image[ndgrid(*rng)] = 1

    return grid_image

def ndgrid(*args, **kwargs):
    """
    Disclaimer: This code is taken directly from the scitools package [1]
    Since at the time of writing scitools predominantly requires python 2.7 while we work with 3.5+
    To avoid issues, we copy the quick code here.
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args, **kwargs)

grid=bw_grid([160,192,224],8)[np.newaxis, ..., np.newaxis]

#--------------------------------------------------- Multipale label dice scores function 
import scipy.io as sio
good_labels = sio.loadmat('../data/labels.mat')['labels'][0]
def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)
        
        
#-----------------------------------------Brain Extraction--------------------

def brain_extractor(brain):
    ext = Extractor()
    prob = ext.run(brain) 
    # mask can be obtained as:
    mask = prob > 0.7
    image=brain*mask
    return image
    
def imgnorm(N_I,index1=0.0001,index2=0.0001):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1*len(I_sort))]
    I_max = I_sort[-int(index2*len(I_sort))]
    
    N_I =1.0*(N_I-I_min)/(I_max-I_min)
    N_I[N_I>1.0]=1.0
    N_I[N_I<0.0]=0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2
def Norm_Zscore(img):
    img= (img-np.mean(img))/np.std(img) 
    return img
        


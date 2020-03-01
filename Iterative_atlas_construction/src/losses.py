"""
losses for VoxelMorph
"""


# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np
import torch
import sys
sys.path.append('/data/My_Thesis/Code/new_code/voxelmorph/ext/neuron')
sys.path.append('/data/My_Thesis/Code/new_code/voxelmorph/ext/pynd-lib')
sys.path.append('/data/My_Thesis/Code/new_code/voxelmorph/ext/pytools-lib')
#from  neuron1 import layers as nrn
import neuron.layers as nrn_layers
import neuron.utils as nrn_utils
11
def binary_dice(y_true, y_pred):
    """
    N-D dice for binary segmentation
    """
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = 1 + np.range(ndims)

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice


class NCC():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps


    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):
        return - self.ncc(I, J)


class Grad():
    """
    N-D gradient loss
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]
            
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)
        
        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            df = [tf.reduce_mean(f * f) for f in self._diffs(y_pred)]
        return tf.add_n(df) / len(df)


class Miccai2018():
    """
    N-D main loss for VoxelMorph MICCAI Paper
    prior matching (KL) term + image matching term
    """

    def __init__(self, image_sigma, prior_lambda, flow_vol_shape=None):
        self.image_sigma = image_sigma
        self.prior_lambda = prior_lambda
        self.D = None
        self.flow_vol_shape = flow_vol_shape


    def _adj_filt(self, ndims):
        """
        compute an adjacency filter that, for each feature independently, 
        has a '1' in the immediate neighbor, and 0 elsewehre.
        so for each filter, the filter has 2^ndims 1s.
        the filter is then setup such that feature i outputs only to feature i
        """

        # inner filter, that is 3x3x...
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied 
        # ith feature to ith feature
        filt = np.zeros([3] * ndims + [ndims, ndims])
        for i in range(ndims):
            filt[..., i, i] = filt_inner
                    
        return filt


    def _degree_matrix(self, vol_shape):
        # get shape stats
        ndims = len(vol_shape)
        sz = [*vol_shape, ndims]

        # prepare conv kernel
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # prepare tf filter
        z = K.ones([1] + sz)
        filt_tf = tf.convert_to_tensor(self._adj_filt(ndims), dtype=tf.float32)
        strides = [1] * (ndims + 2)
        return conv_fn(z, filt_tf, strides, "SAME")


    def prec_loss(self, y_pred):
        """
        a more manual implementation of the precision matrix term
                mu * P * mu    where    P = D - A
        where D is the degree matrix and A is the adjacency matrix
                mu * P * mu = 0.5 * sum_i mu_i sum_j (mu_i - mu_j) = 0.5 * sum_i,j (mu_i - mu_j) ^ 2
        where j are neighbors of i

        Note: could probably do with a difference filter, 
        but the edges would be complicated unless tensorflow allowed for edge copying
        """
        vol_shape = y_pred.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)
        
        sm = 0
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y_pred, r)
            df = y[1:, ...] - y[:-1, ...]
            sm += K.mean(df * df)

        return 0.5 * sm / ndims


    def kl_loss(self, y_true, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3

        y_true is only used to get the shape
        """

        # prepare inputs
        ndims = len(y_pred.get_shape()) - 2
        mean = y_pred[..., 0:ndims]
        log_sigma = y_pred[..., ndims:]
        if self.flow_vol_shape is None:
            # Note: this might not work in multi_gpu mode if vol_shape is not apriori passed in
            self.flow_vol_shape = y_true.get_shape().as_list()[1:-1]

        # compute the degree matrix (only needs to be done once)
        # we usually can't compute this until we know the ndims, 
        # which is a function of the data
        if self.D is None:
            self.D = self._degree_matrix(self.flow_vol_shape)

        # sigma terms
        sigma_term = self.prior_lambda * self.D * tf.exp(log_sigma) - log_sigma
        sigma_term = K.mean(sigma_term)

        # precision terms
        # note needs 0.5 twice, one here (inside self.prec_loss), one below
        prec_term = self.prior_lambda * self.prec_loss(mean)

        # combine terms
        return 0.5 * ndims * (sigma_term + prec_term) # ndims because we averaged over dimensions as well

        
    def _MINDSSC3D(self,vol,kernel_hw=2,d=400):
        if isinstance(vol.shape, (tf.Dimension, tf.TensorShape)):
            volshape = vol.shape.as_list()
        else:
            volshape = vol.shape

        H = volshape[0]; W = volshape[1]; D = volshape[2]
        
        vol=final_data=tf.reshape(data,(H,W,D,1))
        
        #define spatial offset layout for 12 self-similarity patches
        H = volshape[0];
        W = volshape[1];
        D = volshape[2]

        vol = final_data = tf.reshape(data, (H, W, D, 1))

    # define spatial offset layout for 12 self-similarity patches
        delta = 1.5
    # H = 160
    # W = 192
    # D = 224
        H = volshape[0]; W = volshape[1]; D = volshape[2]
        #define spatial offset layout for 12 self-similarity patches
        theta_ssc = np.zeros((2,12,3))
        theta_ssc[0,:,0] = np.array([ 0, 0,-d,-d, 0, 0, 0, 0,+d,+d, 0, 0])/D
        theta_ssc[0,:,1] = np.array([ 0, 0, 0, 0,+d,+d,-d,-d, 0, 0, 0, 0])/W
        theta_ssc[0,:,2] = np.array([-d,-d, 0, 0, 0, 0, 0, 0, 0, 0,+d,+d])/H
        theta_ssc[1,:,0] = np.array([-d, 0, 0, 0,-d,+d, 0,+d, 0, 0, 0, 0])/D
        theta_ssc[1,:,1] = np.array([ 0,+d,-d, 0, 0, 0, 0, 0, 0, 0,-d,+d])/W
        theta_ssc[1,:,2] = np.array([ 0, 0, 0,+d, 0, 0,-d, 0,-d,+d, 0, 0])/H

            # computing the shifting by filter
        sampledd = []
        for i in range(12):
            shift = np.zeros((160, 192, 224, 3))
            shift = shift[np.newaxis, ...]
            theta = theta_ssc
            theta11 = theta[0, :, np.newaxis, np.newaxis, np.newaxis, :]
            theta = shift + theta11
            field = tf.convert_to_tensor(theta[i, ...], dtype=tf.float32)
            my_sampled = nrn_utils.transform(vol, field)
                    # .......................
            theta1 = theta_ssc
            theta1 = theta1[1, :, np.newaxis, np.newaxis, np.newaxis, :]
            theta1 = shift + theta1
            field = tf.convert_to_tensor(theta1[i, ...], dtype=tf.float32)
            my_sampled1 = nrn_utils.transform(vol, field)
            sampledd.append(my_sampled - my_sampled1)
        sampledd = tf.stack(sampledd)
        sampledd = tf.transpose(sampledd, [4, 1, 2, 3, 0])

        my_mind = tf.keras.layers.AveragePooling3D(pool_size=(kernel, kernel, kernel), strides=(1, 1, 1), padding='same',data_format='channels_last')(tf.abs(sampledd) ** 2)
        my_mind -= tf.keras.backend.min(my_mind, axis=-1, keepdims=True)
        my_mind /= (tf.keras.backend.sum(my_mind, axis=-1, keepdims=True) + 0.001)
        return tf.keras.backend.exp(-my_mind)
        
    def recon_loss(self, y_true, y_pred):
        
        #y_true=self._MINDSSC3D(y_true)
        #y_pred=self._MINDSSC3D(y_pred)
        #y_pred= self._MIND(y_pred)
        #y_true = self._MIND(y_true)
        """ reconstruction loss """
        return 1. / (self.image_sigma**2) * K.mean(K.square(y_true - y_pred))
        #return  K.mean(K.square(data_true - data_pred))


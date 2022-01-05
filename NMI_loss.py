import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
class NMI:

    def __init__(self, bin_centers=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],sigma_ratio=0.5, max_clip=1):
        """
        Mutual information loss for image-image pairs.
        Author: Courtney Guo

        If you use this loss function, please cite the following:

        Guo, Courtney K. Multi-modal image registration with unsupervised deep learning. MEng. Thesis

        Unsupervised Learning of Probabilistic Diffeomorphic Registration for Images and Surfaces
        Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
        MedIA: Medial Image Analysis. 2019. eprint arXiv:1903.03545
        """
        print("vxm info: mutual information loss is experimental", file=sys.stderr)
        self.max_clip = max_clip
        self.vol_bin_centers = K.variable(bin_centers)
        self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = K.variable(1 / (2 * np.square(self.sigma)))

    def global_mi(self, y_true, y_pred):
        # reshape: flatten images into shape (batch_size, height*width*channel, 1)
        y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
        y_true = K.expand_dims(y_true, 2)
        y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
        y_pred = K.expand_dims(y_pred, 2)

        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(self.vol_bin_centers.get_shape().as_list())]
        vbc = K.reshape(self.vol_bin_centers, o)

        # compute image terms
        I_a = K.exp(- self.preterm * K.square(y_true  - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- self.preterm * K.square(y_pred  - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a, (0,2,1))
        pab = K.batch_dot(I_a_permute, I_b)
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keepdims=True)
        pb = tf.reduce_mean(I_b, 1, keepdims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0,2,1)), pb) + K.epsilon()
        return K.sum(K.sum(pab * K.log(pab/papb + K.epsilon()), 1), 1)

    def loss(self, y_true, y_pred):
        y_pred = K.clip(y_pred, 0, self.max_clip)
        y_true = K.clip(y_true, 0, self.max_clip)
        return -self.global_mi(y_true, y_pred)


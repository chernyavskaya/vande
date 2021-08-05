import os
#import setGPU
import numpy as np
import tensorflow as tf
from collections import namedtuple
import matplotlib.pyplot as plt

import vae.losses as losses
import vae.vae_base as vbase
import vae.layers as layers
import tensorflow.keras.layers as klayers
from tensorflow import keras
import ADgvae.models.custom_functions as funcs



class VAE_ParticleNet(vbase.VAE):

   def __init__(self, **kwargs):
      super(VAE_ParticleNet, self).__init__(**kwargs)
      self.setting = self.params.setting
      self.with_bn = self.setting.with_bn if self.setting.with_bn!=None else True 
      self.latent_dim = self.setting.latent_dim
      self.activation = self.setting.activation
      self.kl_warmup_time = self.setting.kl_warmup_time
      self.name = self.setting.name
      self.beta_kl_warmup = tf.Variable(0.0, trainable=False, name='beta_kl_warmup', dtype=tf.float32)


   def build_encoder(self, additional_inputs):

        points = klayers.Input(name='points', shape=self.setting.input_shapes['points'])
        features = klayers.Input(name='features', shape=self.setting.input_shapes['features']) if 'features' in self.setting.input_shapes else None
        particlenet_layer = self.build_particlenet(points,features)
        x = particlenet_layer
        if 'vae'.lower() in self.setting.ae_type : #TO DO: add some dense layers in between
            for layer_idx in range(0,len(self.setting.conv_params_encoder)):
                layer_param  = self.setting.conv_params_encoder[layer_idx]
                x = keras.layers.Dense(self.setting.latent_dim*layer_param,activation=self.activation,
                                              kernel_initializer='glorot_normal')(x) 
            self.z_mean = tf.keras.layers.Dense(self.setting.latent_dim, name='z_mean')(x) #no activation
            self.z_log_var = tf.keras.layers.Dense(self.setting.latent_dim, name='z_log_var')(x) #no activation
            # use reparameterization trick to push the sampling out as input
            self.z = layers.Sampling()((self.z_mean, self.z_log_var))
            encoder_output = [self.z,self.z_mean,self.z_log_var]
            encoder_model = tf.keras.Model(inputs=(points,features), outputs=encoder_output,name='encoder')

        else : 
            for layer_idx in range(0,len(self.setting.conv_params_encoder)):
                layer_param  = self.setting.conv_params_encoder[layer_idx]
                x = keras.layers.Dense(self.setting.latent_dim*layer_param,activation=self.activation,
                                              kernel_initializer='glorot_normal')(x) 
            self.z = keras.layers.Dense(self.setting.latent_dim,activation=self.activation,
                                              kernel_initializer='glorot_normal')(x) 
            encoder_output = [self.z]
            encoder_model = tf.keras.Model(inputs=(points,features), outputs=encoder_output,name='encoder')
            
        encoder_model.summary() 
        return encoder_model


   def build_decoder(self, additional_inputs):
        input_layer   = klayers.Input(shape=(self.setting.latent_dim, ), name='decoder_input')
        num_dense_channels = self.setting.conv_params_decoder[0]

        #x = klayers.Dense((self.setting.num_points*num_dense_channels),activation=self.activation )(input_layer)
        #x = klayers.BatchNormalization(name='%s_dense_0' % (self.name))(x)
        x = keras.layers.Dense((self.setting.num_points*num_dense_channels),
                               kernel_initializer='glorot_normal')(input_layer) 
        #TO DO: order of BN->Activation or the other way around can have impact, check
        if self.with_bn:
            x = klayers.BatchNormalization(name='%s_dense_0' % (self.name))(x)
        if self.activation:
            x = klayers.Activation(self.activation, name='%s_act_0' % (self.name))(x)  
        x = klayers.Reshape((self.setting.num_points,num_dense_channels), input_shape=(self.setting.num_points*num_dense_channels,))(x)
 
        for layer_idx in range(1,len(self.setting.conv_params_decoder)):
            layer_param  = self.setting.conv_params_decoder[layer_idx]
            #1D and 2D  Conv layers with kernel and stride side of 1 are identical operations, but for 2D first need to expand then to squeeze
            #x = tf.squeeze(keras.layers.Conv2D(layer_param, kernel_size=(1, 1), strides=1, data_format='channels_last',
            #                        use_bias=False if self.with_bn else True, activation=self.activation, kernel_initializer='glorot_normal',
            #                        name='%s_conv_%d' % (self.name,layer_idx))(tf.expand_dims(x, axis=2)),axis=2)  
            #x = klayers.BatchNormalization(name='%s_bn_%d' % (self.name,layer_idx))(x)
            x = klayers.Conv2D(layer_param, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal',
                                    name='%s_conv_%d' % (self.name,layer_idx))(tf.expand_dims(x, axis=2))
            if self.with_bn:
                x = klayers.BatchNormalization(name='%s_bn_%d' % (self.name,layer_idx))(x)
            x = tf.squeeze(x, axis=2)
            if self.activation:
                x = klayers.Activation(self.activation, name='%s_act_%d' % (self.name,layer_idx))(x)  

        decoder_output = tf.squeeze(klayers.Conv2D(self.setting.num_features, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=True, activation=self.activation, kernel_initializer='glorot_normal',
                                    name='%s_conv_out' % self.name)(tf.expand_dims(x, axis=2)),axis=2) 
        decoder = tf.keras.Model(inputs=input_layer, outputs=decoder_output,name='decoder')
        decoder.summary()
        return decoder 



   def build_edgeconv(self,points,features,K=7,channels=32,name=''):
      """EdgeConv
        K: int, number of neighbors
        in_channels: # of input channels
        channels: tuple of output channels
        pooling: pooling method ('max' or 'average')
    Inputs:
        points: (N, P, C_p)
        features: (N, P, C_0)
    Returns:
        transformed points: (N, P, C_out), C_out = channels[-1]
    """
      with tf.name_scope('EdgeConv_'):        
         # distance
         D = funcs.batch_distance_matrix_general(points, points)  # (N, P, P)
         _, indices = tf.nn.top_k(-D, k=K + 1)  # (N, P, K+1)
         indices = indices[:, :, 1:]  # (N, P, K)

         fts = features
         knn_fts = funcs.knn(self.setting.num_points, K, indices, fts)  # (N, P, K, C)
         knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, K, 1))  # (N, P, K, C)
         #knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C) #TO DO: Investigate why this assymetric function actually performs worse 
         knn_fts =  tf.subtract(knn_fts, knn_fts_center) #TO DO : This edge function should be local info only

         x = knn_fts
         for idx, channel in enumerate(channels):
            x = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                        use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal', name='%s_conv%d' % (name, idx))(x)
            if self.with_bn:
               x = keras.layers.BatchNormalization(name='%s_bn%d' % (name, idx))(x)
            if self.activation:
               x = keras.layers.Activation(self.activation, name='%s_act%d' % (name, idx))(x)

         if self.setting.conv_pooling == 'max':
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
         else:
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')
                
         # shortcut of constituents features
         sc = keras.layers.Conv2D(channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                                     use_bias=False if self.with_bn else True, kernel_initializer='glorot_normal', name='%s_sc_conv' % name)(tf.expand_dims(features, axis=2))
         if self.with_bn:
                sc = keras.layers.BatchNormalization(name='%s_sc_bn' % name)(sc)
         sc = tf.squeeze(sc, axis=2)

         x = fts #TO DO : investigate why inclusing sum/concatenation degrades performance. In fact sensitivity to different signals is removed
         #Right now there is no use of shortcut         

          
       #  x = sc + fts #sum by default, original PN. It probably should be added after latent space
       #  if self.setting.conv_linking == 'concat': #concat or sum
       #     x = tf.concat([sc,fts],axis=2) 
       #  if self.activation:
       #     x =  keras.layers.Activation(self.activation, name='%s_sc_act' % name)(x)  # (N, P, C') #TO DO : try with concatenation instead of sum
         return x



   def build_particlenet(self,points, features):
        with tf.name_scope('ParticleNetBase'):

           #mask = keras.Input(name='mask', shape=self.setting.input_shapes['mask']) if 'mask' in self.setting.input_shapes else None
           mask = None #TO DO : need to check how to implement that when/if we need it

           if mask is not None:
               mask = tf.cast(tf.not_equal(mask, 0), dtype='float32')  # 1 if valid
               coord_shift = tf.multiply(999., tf.cast(tf.equal(mask, 0), dtype='float32'))  # make non-valid positions to 99

           if self.with_bn:
               fts = tf.squeeze(klayers.BatchNormalization(name='%s_fts_bn' % self.name)(tf.expand_dims(features, axis=2)), axis=2)
           fts = features 
           for layer_idx, layer_param in enumerate(self.setting.conv_params):
               K, channels = layer_param
               if mask is not None:
                   pts = tf.add(coord_shift, points) if layer_idx == 0 else tf.add(coord_shift, fts)
               else : pts=points
               fts_shape = fts.get_shape().as_list()
               pts_shape = pts.get_shape().as_list()
               fts = self.build_edgeconv(pts,fts,K=K,channels=channels,name='%s_%i'%(self.name,layer_idx))

           if mask is not None:
               fts = tf.multiply(fts, mask)

           #pool = tf.reduce_mean(fts, axis=1)  # (N, C)  #pooling over all jet constituents
           # Flatten to format for MLP input
           pool=klayers.Flatten(name='Flatten_PN')(fts) #seems liks Flatteneing or pooling performs the same, interesting because when flatten one has much more parameters...

           return pool


   @classmethod
   def load(cls, path):
      custom_objects = {'Sampling': layers.Sampling}
      return super().load(path=path, custom_objects=custom_objects)

           
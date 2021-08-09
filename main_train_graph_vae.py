import sys,os
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../sarewt_orig/')))
sys.path.append(os.path.abspath(os.path.join('../../')))
import setGPU
import numpy as np
from collections import namedtuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
print('tensorflow version: ', tf.__version__)
import tensorflow.keras.layers as klayers

import vae.vae_particlenet as vae_pn
import vae.losses as losses
import pofah.path_constants.sample_dict_file_parts_input_vae as sdi
import pofah.util.experiment as expe
import pofah.util.sample_factory as safa
import util.data_generator_particlenet as dage_pn
import sarewt.data_reader as dare
import pofah.phase_space.cut_constants as cuts
import training as tra
import time


# ********************************************************
#       runtime params
# ********************************************************
Parameters = namedtuple('Parameters', 'run_n input_shape kernel_sz kernel_ini_n beta epochs train_total_n gen_part_n valid_total_n batch_n z_sz activation initializer learning_rate max_lr_decay lambda_reg')
params = Parameters(run_n=1, 
                    input_shape=[(100,2),(100,3)],
                    beta=0.1, 
                    epochs=2, 
                    train_total_n=int(7e3), 
                    valid_total_n=int(7e3), 
                    gen_part_n=int(2e3), 
                    batch_n=256, 
                    z_sz=12, #latent dimension
                    activation=klayers.LeakyReLU(alpha=0.1),
                    learning_rate=0.001,
                    max_lr_decay=8, 
                    lambda_reg=0.0, # 'L1L2'
                    initializer='he_uniform', #TO DO :incorporate to particle net

                    kernel_sz=(1,3),  #TODO : not used but needed 
                    kernel_ini_n=12 #TODO : not used but needed 
                    )

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)

class _DotDict:
    pass

nodes_n = params.input_shape[1][0]
feat_sz = params.input_shape[1][1]

setting = _DotDict()
setting.name = 'PN'
 # conv_params: list of tuple in the format (K, (C1, C2, C3))
setting.conv_params = [
   (15, ([20,20,20])),
  #      (7, (32, 32, 32)),
  #      (7, (64, 64, 64)),
        ]
#setting.conv_params_decoder = [60,32,16,8, 5]
setting.conv_params_encoder_input = 20 #64 #20
setting.conv_params_encoder = []
setting.conv_params_decoder = [10]  #[32,16,8]
setting.with_bn = True
# conv_pooling: 'average' or 'max' #indeed average seems to perform better
setting.conv_pooling = 'average'
setting.conv_linking = 'sum' #concat or sum
setting.num_points = nodes_n #num of original consituents
setting.num_features = feat_sz #num of original features
setting.input_shapes = {'points': [nodes_n,feat_sz-1],'features':[nodes_n,feat_sz]}
setting.latent_dim = params.z_sz
setting.ae_type = 'vae'  #ae or vae 
setting.kl_warmup_time = 0 
setting.activation = params.activation 

# ********************************************************
#       prepare training (generator) and validation data
# ********************************************************

# train (generator)
print('>>> Preparing training dataset generator')


data_train_generator = dage_pn.DataGeneratorDirect(path=paths.sample_dir_path('qcdSide'), 
                                                   sample_part_n=params.gen_part_n, 
                                                   sample_max_n=params.train_total_n, 
                                                   batch_size = params.batch_n,
                                                   **cuts.global_cuts
                                                  ) # generate 10 M jet samples
train_ds = tf.data.Dataset.from_generator(data_train_generator, 
                                         (tf.float32,tf.float32), 
                                         (tf.TensorShape([None,params.input_shape[0][0],params.input_shape[0][1]]),
                                          tf.TensorShape([None,params.input_shape[1][0],params.input_shape[1][1]]))
                                         )
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# validation (full tensor, 1M events -> 2M samples)
print('>>> Preparing validation dataset')
const_valid, _, features_valid, _ = dare.DataReader(path=paths.sample_dir_path('qcdSideExt')).read_events_from_dir(read_n=params.valid_total_n, **cuts.global_cuts)
data_valid = dage_pn.events_to_input_samples(const_valid, features_valid)
data_valid = dage_pn.normalize_features(data_valid)
valid_ds = tf.data.Dataset.from_tensor_slices((data_valid[:,:,0:2],data_valid[:,:,:])).batch(params.batch_n, drop_remainder=True)

# *******************************************************
#                       training options
# *******************************************************

optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
loss_fn = losses.threeD_loss

# *******************************************************
#                       build model
# *******************************************************
vae = vae_pn.VAE_ParticleNet(beta=params.beta,setting=setting,kernel_ini_n=params.kernel_ini_n,name='PN_AE_',input_shape=params.input_shape)
a = [1] #TODO : fix this hack
vae.build(a)

# *******************************************************
#                       train and save
# *******************************************************
print('>>> Launching Training')
start_time = time.time()

trainer = tra.TrainerParticleNet(optimizer=optimizer, beta=params.beta, patience=3, min_delta=0.03, max_lr_decay=params.max_lr_decay, lambda_reg=params.lambda_reg)
losses_reco, losses_valid = trainer.train(vae=vae, loss_fn=loss_fn,
                                          train_ds=train_ds,valid_ds=valid_ds,
                                          epochs=params.epochs, model_dir=experiment.model_dir)

end_time = time.time()
print(f"Runtime of the training is {end_time - start_time}")

tra.plot_training_results(losses_reco, losses_valid, experiment.fig_dir)

vae.save(path=experiment.model_dir)

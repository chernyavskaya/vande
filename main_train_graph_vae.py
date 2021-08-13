import sys,os
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../sarewt_orig/')))
sys.path.append(os.path.abspath(os.path.join('../../')))
import glob
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
import h5py
import json
import matplotlib
matplotlib.use('Agg')


# ********************************************************
#       runtime params
# ********************************************************
RunParameters = namedtuple('Parameters', 'run_n  \
 epochs train_total_n gen_part_n valid_total_n batch_n learning_rate max_lr_decay lambda_reg generator')
params = RunParameters(run_n=12, 
                       epochs=20, 
                       train_total_n=int(5e5 ),  #2e6 
                       valid_total_n=int(1e5), #1e5
                       gen_part_n=int(1e5), #1e5
                       batch_n=256, 
                       learning_rate=0.0001,
                       max_lr_decay=5, 
                       lambda_reg=0.0,# 'L1L2'
                       generator=0)  #run generator or not

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)

# ********************************************************
#       Models params
# ********************************************************
Parameters = namedtuple('Settings', 'name  input_shape  beta activation initializer conv_params conv_params_encoder conv_params_decoder with_bn conv_pooling conv_linking latent_dim ae_type kl_warmup_time kernel_ini_n')
settings = Parameters(name = 'PN',
                     input_shape=[(100,2),(100,3)],
                     beta=0.01, 
                     activation=klayers.LeakyReLU(alpha=0.1),
                     initializer='glorot_normal', 
                      # conv_params: list of tuple in the format (K, (C1, C2, C3))
                     conv_params = [(7, (32, 32, 32)),
                                    (7, (64, 64, 64)),
                                    ],
                                #   [(16, (64, 64, 64)),
                                #    (16, (128, 128, 128)),
                                #    (16, (256, 256, 256)),
                                #   ],
                     conv_params_encoder = [20],
                     conv_params_decoder = [50,30,10,5],  #[32,16,8]
                     with_bn = True,
                     conv_pooling = 'average',
                     conv_linking = 'sum' ,#concat or sum #currently shorcut is removed
                     latent_dim = 5,
                     ae_type = 'vae',  #ae or vae 
                     kl_warmup_time = 10, #currently noy used
                     kernel_ini_n = 0) #should be/will be removed at next iteration


''' saving model parameters''' 
SetupParameters = namedtuple("SetupParameters", RunParameters._fields + Parameters._fields)
save_params = SetupParameters(*(params + settings))
saev_params_json = json.dumps((save_params._replace(activation='activation'))._asdict()) #replacing activation as you cannot save it
with open(os.path.join(experiment.model_dir,'parameters.json'), 'w', encoding='utf-8') as f_json:
    json.dump(saev_params_json, f_json, ensure_ascii=False, indent=4)

# ********************************************************
#       prepare training (generator) and validation data
# ********************************************************
print('>>> Launching Training')
start_time = time.time()
# train (generator)
if params.generator:
    print('>>> Preparing training dataset generator')
    data_train_generator = dage_pn.DataGeneratorDirect(path=paths.sample_dir_path('qcdSide'), 
                                                       sample_part_n=params.gen_part_n, 
                                                       sample_max_n=params.train_total_n, 
                                                       batch_size = params.batch_n,
                                                       **cuts.global_cuts
                                                      ) # generate 10 M jet samples
    train_ds = tf.data.Dataset.from_generator(data_train_generator, 
                                             (tf.float32,tf.float32), 
                                             (tf.TensorShape([None,settings.input_shape[0][0],settings.input_shape[0][1]]),
                                              tf.TensorShape([None,settings.input_shape[1][0],settings.input_shape[1][1]]))
                                             )
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
else :
    # training (full tensor, 1M events -> 2M samples)
    print('>>> Preparing training dataset')
    #const_train, _, features_train, _ = dare.DataReader(path=paths.sample_dir_path('qcdSide')).read_events_from_dir(read_n=params.train_total_n, **cuts.global_cuts)
    data_train = dage_pn.get_data_from_file(path=paths.sample_dir_path('qcdSide'),file_num=0,end=params.train_total_n)
    train_ds = tf.data.Dataset.from_tensor_slices((data_train[:,:,0:2],data_train[:,:,:])).batch(params.batch_n, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
     #TODO : it has to be shuffled! right now it is shuffled only inside generator
    
# validation (full tensor, 1M events -> 2M samples)
print('>>> Preparing validation dataset')
#const_valid, _, features_valid, _ = dare.DataReader(path=paths.sample_dir_path('qcdSideExt')).read_events_from_dir(read_n=params.valid_total_n, **cuts.global_cuts)
data_valid = dage_pn.get_data_from_file(path=paths.sample_dir_path('qcdSideExt'),file_num=0,end=params.valid_total_n)
valid_ds = tf.data.Dataset.from_tensor_slices((data_valid[:,:,0:2],data_valid[:,:,:])).batch(params.batch_n, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# *******************************************************
#                       training options
# *******************************************************

optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
loss_fn = losses.threeD_loss

# *******************************************************
#                       build model
# *******************************************************
vae = vae_pn.VAE_ParticleNet(name=settings.name,conv_params=settings.conv_params, conv_params_encoder=settings.conv_params_encoder,
                                            conv_params_decoder=settings.conv_params_decoder, with_bn=settings.conv_params_decoder, 
                                            conv_pooling=settings.conv_pooling,conv_linking=settings.conv_linking,
                                            initializer=settings.initializer,
                                            input_shape=settings.input_shape,latent_dim=settings.latent_dim,ae_type=settings.ae_type,
                                            kl_warmup_time=settings.kl_warmup_time,activation=settings.activation,beta=settings.beta )
vae.build()
#vae.save(path=os.path.join(experiment.model_dir,'not_trained_model'))

# *******************************************************
#                       train and save
# *******************************************************


trainer = tra.TrainerParticleNet(optimizer=optimizer, beta=settings.beta, patience=3, min_delta=0.03, max_lr_decay=params.max_lr_decay, lambda_reg=params.lambda_reg)
losses_train, losses_valid = trainer.train(vae=vae, loss_fn=loss_fn,
                                          train_ds=train_ds,valid_ds=valid_ds,
                                          epochs=params.epochs, model_dir=experiment.model_dir)

end_time = time.time()
print(f"Runtime of the training is {end_time - start_time}")
vae.save(path=experiment.model_dir)

for i in range(len(losses_train)):
    loss_train,loss_valid =  losses_train[i],losses_valid[i]
    tra.plot_training_results(loss_train, loss_valid, experiment.model_dir,trainer.saved_loss_types[i])


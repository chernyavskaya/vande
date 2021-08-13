import sys,os, glob
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
from vae.vae_particlenet import VAE_ParticleNet
import pofah.path_constants.sample_dict_file_parts_input_vae as sdi
import pofah.util.experiment as expe
import pofah.util.sample_factory as safa
import util.data_generator_particlenet as dage_pn
import sarewt.data_reader as dare
import pofah.phase_space.cut_constants as cuts
import training as tra
import time,pathlib, h5py, json, matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('/eos/user/n/nchernya/MLHEP/AnomalyDetection/ADgvae/utils/adfigstyle.mplstyle')
import util.util_plotting as plot
import vande.analysis.analysis_roc as ar

# ********************************************************
#       runtime params
# ********************************************************
RunParameters = namedtuple('Parameters', 'run_n test_total_n batch_n')
params = RunParameters(run_n=7, test_total_n=int(5e4), batch_n=500)  #times 2, because two jets  

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)


# ********************************************************
#       prepare testing data
# ********************************************************


print('>>> Preparing testing BG dataset')
particles_dict = {}
particles_dict['input_feats'] = {} # dataset prepared for inference
data_test = dage_pn.get_data_from_file(path=paths.sample_dir_path('qcdSideExt'),file_num=-1,end=params.test_total_n)
particles_dict['input_feats']['BG'] = tf.data.Dataset.from_tensor_slices((data_test[:,:,0:2],data_test[:,:,:])).batch(params.batch_n, drop_remainder=True).prefetch(buffer_size=1)

sig_types = 'GtoWW35na,GtoWW15na,GtoWW35br,GtoWW15br'.split(',')
for sig in sig_types:
    data_test_sig = dage_pn.get_data_from_file(path=paths.sample_dir_path(sig),file_num=-1,end=params.test_total_n)
    sig_ds = tf.data.Dataset.from_tensor_slices((data_test_sig[:,:,0:2],data_test_sig[:,:,:])).batch(params.batch_n, drop_remainder=True).prefetch(buffer_size=1)
    particles_dict['input_feats'][sig]= sig_ds


# *******************************************************
#                       losses options
# *******************************************************

loss_fn = losses.threeD_loss

# *******************************************************
#                       load model
# *******************************************************
vae = VAE_ParticleNet.from_saved_model(path=experiment.model_dir)
beta= vae.beta

# *******************************************************
#                       predicting 
# *******************************************************
for key in 'reco_feats,loss_reco,loss_kl,loss_tot'.split(','):
    particles_dict[key] = {}
particles_dict['reco_feats']['BG'], particles_dict['loss_reco']['BG'], particles_dict['loss_kl']['BG'] = tra.predict_particle_net(vae.model, loss_fn, particles_dict['input_feats']['BG'])
particles_dict['loss_tot']['BG'] = particles_dict['loss_reco']['BG']+beta*particles_dict['loss_kl']['BG']

for sig in sig_types:
    particles_dict['reco_feats'][sig], particles_dict['loss_reco'][sig], particles_dict['loss_kl'][sig] = tra.predict_particle_net(vae.model, loss_fn, particles_dict['input_feats'][sig])
    particles_dict['loss_tot'][sig] = particles_dict['loss_reco'][sig]+beta*particles_dict['loss_kl'][sig]


# *******************************************************
#                       plotting losses
# *******************************************************
fig_dir = os.path.join(experiment.model_dir, 'figs/')
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
for loss in ['loss_tot','loss_kl','loss_reco']:
     datas = []
     datas.append(particles_dict[loss]['BG'])
     for sig in sig_types:
         datas.append(particles_dict[loss][sig])
     plot.plot_hist_many(datas, loss.replace('_',' ') ,'Normalized Dist.' , 'run_n={}'.format(params.run_n), plotname='{}plot_{}'.format(fig_dir,loss), legend=['BG']+sig_types, ylogscale=True)

# *******************************************************
#                       plotting ROCs 
# *******************************************************
for loss in ['loss_tot','loss_kl','loss_reco']:
    neg_class_losses = [particles_dict[loss]['BG']]*len(sig_types)
    pos_class_losses = []
    for sig in sig_types:
       pos_class_losses.append(particles_dict[loss][sig])
    ar.plot_roc( neg_class_losses, pos_class_losses, legend=sig_types, title='run_n={},from {}'.format(params.run_n,loss.replace('_',' ')),
            plot_name='ROC_{}'.format(loss), fig_dir=fig_dir,log_x=False )


# *******************************************************
#                       plotting reco features 
# *******************************************************

#plot features separately for BG, for SIG, and for BG and only one sig together, not to overload the relevant plots


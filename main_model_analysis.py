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
tfds = tf.data.Dataset

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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ********************************************************
#       runtime params
# ********************************************************
RunParameters = namedtuple('Parameters', 'run_n test_total_n batch_n')
params = RunParameters(run_n=13, test_total_n=int(1e5), batch_n=256)  #number of test events is times 2, because two jets  

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)


# ********************************************************
#       prepare testing data
# ********************************************************


print('>>> Preparing testing BG dataset')
particles_dict = {}
particles_dict['input_ds'] = {} # dataset prepared for inference
particles_dict['input_feats'] = {} # input features
particles_dict['latent_space'] = {} 
data_test = dage_pn.get_data_from_file(path=paths.sample_dir_path('qcdSideExt'),file_num=-1,end=params.test_total_n)
particles_dict['input_ds']['BG'] = tf.data.Dataset.from_tensor_slices((data_test[:,:,0:2],data_test[:,:,:])).batch(params.batch_n, drop_remainder=True).prefetch(buffer_size=1)
particles_dict['input_feats']['BG']= data_test

print('>>> Preparing testing signals dataset')
#sig_types = 'GtoWW35na,GtoWW15na,GtoWW35br,GtoWW15br'.split(',')
sig_types = ['GtoWW35na']
for sig in sig_types:
    data_test_sig = dage_pn.get_data_from_file(path=paths.sample_dir_path(sig),file_num=-1,end=params.test_total_n)
    sig_ds = tf.data.Dataset.from_tensor_slices((data_test_sig[:,:,0:2],data_test_sig[:,:,:])).batch(params.batch_n, drop_remainder=True).prefetch(buffer_size=1)
    particles_dict['input_ds'][sig]= sig_ds
    particles_dict['input_feats'][sig]= data_test_sig 


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
print('Predicting')
for key in 'reco_feats,loss_reco,loss_kl,loss_tot'.split(','):
    particles_dict[key] = {}
particles_dict['reco_feats']['BG'], particles_dict['loss_reco']['BG'], particles_dict['loss_kl']['BG'],particles_dict['latent_space']['BG'] = tra.predict_particle_net(vae, loss_fn, particles_dict['input_ds']['BG'],return_latent=True)
#From Run 15 : it should be (1-beta)+beta 
particles_dict['loss_tot']['BG'] = particles_dict['loss_reco']['BG']+beta*particles_dict['loss_kl']['BG']

for sig in sig_types:
    particles_dict['reco_feats'][sig], particles_dict['loss_reco'][sig], particles_dict['loss_kl'][sig],particles_dict['latent_space'][sig] = tra.predict_particle_net(vae, loss_fn, particles_dict['input_ds'][sig],return_latent=True)
#From Run 15 : it should be (1-beta)+beta 
    particles_dict['loss_tot'][sig] = particles_dict['loss_reco'][sig]+beta*particles_dict['loss_kl'][sig]


# *******************************************************
#                       plotting losses
# *******************************************************
print('Plotting losses')
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
print('Plotting ROCs')
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
#TO DO :  Residuals does not make sense to plot for graph architecture? 
#plot features separately for BG, for SIG, and for BG and only one sig together, not to overload the relevant plots
print('Plotting features')
reco_feats = []
num_feats = particles_dict['input_feats']['BG'].shape[-1]
reco_feats.append(particles_dict['input_feats']['BG'].reshape(-1, num_feats))
reco_feats.append(particles_dict['reco_feats']['BG'].reshape(-1, num_feats))
plot.plot_features(reco_feats, 'Input vs Reco.' ,'Normalized Dist.' , 'run_n={}'.format(params.run_n), plotname='{}plot_features_{}'.format(fig_dir,'BG'), legend=('Input {},Reco. {}'.format('BG','BG')).split(','), ylogscale=True)

reco_feats.append(particles_dict['input_feats'][sig_types[0]].reshape(-1, num_feats))
reco_feats.append(particles_dict['reco_feats'][sig_types[0]].reshape(-1, num_feats))
plot.plot_features(reco_feats, 'input vs reco.' ,'Normalized Dist.' , 'run_n={}'.format(params.run_n), plotname='{}plot_features_{}'.format(fig_dir,'BG_SIG'), legend=('Input {},Reco. {}'.format('BG','BG')).split(',')+('Input {},Reco. {}'.format(sig_types[0],sig_types[0])).split(','), ylogscale=True)


reco_feats = [] #renew to plot reco signals only
for sig in sig_types:
    reco_feats.append(particles_dict['reco_feats'][sig].reshape(-1, num_feats))
plot.plot_features(reco_feats, 'Reco.' ,'Normalized Dist.' , 'run_n={}'.format(params.run_n), plotname='{}plot_features_{}'.format(fig_dir,'reco_SIG'), legend=sig_types, ylogscale=True)

# *******************************************************
#                       plotting latent space : z, z_mean, z_std
# *******************************************************
print('Plotting latent space')

for i_num,var in enumerate(['z_latent','z_mean','z_std']):
    latent_space = []
    latent_space.append(particles_dict['latent_space']['BG'][i_num])
    latent_space.append(particles_dict['latent_space'][sig_types[0]][i_num]) #first type of signal only , for comparison
    plot.plot_features(latent_space, var ,'Normalized Dist.' , 'run_n={}'.format(params.run_n), plotname='{}plot_latent_{}_{}'.format(fig_dir,var,'BG_SIG'), legend=('BG,{}'.format(sig_types[0])).split(','), ylogscale=True)


for i_num,var in enumerate(['z_latent','z_mean','z_std']):
    latent_space = []
    for sig in sig_types:
        latent_space.append(particles_dict['latent_space'][sig][i_num])
    plot.plot_features(latent_space, var ,'Normalized Dist.' , 'run_n={}'.format(params.run_n), plotname='{}plot_latent_{}_{}'.format(fig_dir,var,'SIG'), legend=sig_types, ylogscale=True)


# *******************************************************
#                       PCA / tSNE (visualization only) 
# *******************************************************
print('PCA on latent space')

PCA_n = 2
for i_num,var in enumerate(['z_latent','z_mean','z_std']):
    pca = PCA(n_components=PCA_n)
    pca.fit(particles_dict['latent_space']['BG'][i_num])
    print('{} : Explained variation per principal component in BG: {}'.format(var,pca.explained_variance_ratio_))
    latent_pca_bg_sigs = []
    for data_type in ['BG']+sig_types:
       latent_pca = pca.transform(particles_dict['latent_space'][data_type][i_num])
       latent_pca_bg_sigs.append(latent_pca)
    if PCA_n==2: plot.plot_scatter_many(latent_pca_bg_sigs, '{} PCA #0'.format(var.replace('_',' ')) ,'{} PCA #1'.format(var.replace('_',' ')) , 'run_n={}'.format(params.run_n), plotname='{}plot_PCA_{}_{}'.format(fig_dir,var.replace('_',' '),'BG_SIG'), legend=['BG']+sig_types)
    if PCA_n==1: plot.plot_hist_many(latent_pca_bg_sigs, '{} PCA #0'.format(var.replace('_',' ')) ,'{} PCA #1'.format(var.replace('_',' ')) , 'run_n={}'.format(params.run_n), plotname='{}plot_hist_PCA_{}_{}'.format(fig_dir,var.replace('_',' '),'BG_SIG'), legend=['BG']+sig_types)







   
   

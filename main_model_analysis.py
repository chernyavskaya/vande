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
RunParameters = namedtuple('Parameters', 'run_n  ae_type test_total_n batch_n')
params = RunParameters(run_n=42, ae_type='ae', test_total_n=int(5e4), batch_n=256)  #number of test events is times 2, because two jets  

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)
results_dir = os.path.join(experiment.model_dir, 'predicted/')
pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)


# *******************************************************
#                       load model
# *******************************************************
vae = VAE_ParticleNet.from_saved_model(path=experiment.model_dir)
#vae = VAE_ParticleNet.from_saved_model(path=os.path.join(experiment.model_dir, 'best_so_far/'))
beta= vae.beta
print('Beta of the model is {}'.format(beta))
loss_fn = losses.threeD_loss


# ********************************************************
#       prepare testing data
# ********************************************************

particles_dict = {}
for key in 'input_ds,input_feats,reco_feats,loss_reco,loss_kl,loss_tot,latent_space'.split(','):
    particles_dict[key] = {}
latent_vars=['z_latent']
if 'vae'.lower() in params.ae_type:
    latent_vars+=['z_mean','z_std']
for var in latent_vars:
    particles_dict[var] = {}

print('>>> Preparing testing BG signals dataset')
sig_types = 'GtoWW35na,GtoWW15na,GtoWW35br,GtoWW15br,qcdSideExt'.split(',')
#sig_types = 'GtoWW15na,GtoWW35br'.split(',')

for sig in ['BG']+sig_types:
    sample_name = 'qcdSig' if  'BG' in sig else sig
    out_file_name = '{}{}_njets_{}.h5'.format(results_dir,sample_name,params.test_total_n)
    if pathlib.Path(out_file_name).is_file():
        print('Loading already predicted data')
        with h5py.File(out_file_name, 'r') as inFile:
            for key in particles_dict.keys():
                if key=='latent_space' or key=='input_ds': continue
                particles_dict[key][sig] = np.array(inFile[key])#[0:params.test_total_n,:,:])
    else:
        print('Predicting')
        data_test_sig = dage_pn.get_data_from_file(path=paths.sample_dir_path(sample_name),file_num=-1,end=params.test_total_n,shuffle=False)
        sig_ds = tf.data.Dataset.from_tensor_slices((data_test_sig[:,:,0:2],data_test_sig[:,:,:])).batch(params.batch_n, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        particles_dict['input_ds'][sig]= sig_ds
        particles_dict['input_feats'][sig]= data_test_sig 
        particles_dict['reco_feats'][sig], particles_dict['loss_reco'][sig], particles_dict['loss_kl'][sig],particles_dict['latent_space'][sig] = tra.predict_particle_net(vae, loss_fn, particles_dict['input_ds'][sig],return_latent=True)
        particles_dict['z_latent'][sig] = particles_dict['latent_space'][sig][0]
        if 'vae'.lower() in params.ae_type:
            particles_dict['z_mean'][sig] = particles_dict['latent_space'][sig][1]
            particles_dict['z_std'][sig] = particles_dict['latent_space'][sig][2]
        #From Run 15 : it should be (1-beta)+beta 
        particles_dict['loss_tot'][sig] = (1-beta)*particles_dict['loss_reco'][sig]+beta*particles_dict['loss_kl'][sig]
      # *******************************************************
      #               write predicted data
      # *******************************************************
        print('writing results for {} to {}'.format(sample_name, results_dir))
        with h5py.File(out_file_name, 'w') as outFile:
            for key in particles_dict.keys():
                if key=='latent_space' or key=='input_ds': continue
                else : outFile.create_dataset(key, data=np.array(particles_dict[key][sig], dtype=np.float32), compression='gzip')





# *******************************************************
#                       plotting losses
# *******************************************************
losses = ['loss_reco'] 
if 'vae'.lower() in params.ae_type:
    losses+=['loss_tot','loss_kl']

print('Plotting losses')
fig_dir = os.path.join(experiment.model_dir, 'figs/')
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

for loss in losses:
     datas = []
     datas.append(particles_dict[loss]['BG'])
     for sig in sig_types:
         datas.append(particles_dict[loss][sig])
     plot.plot_hist_many(datas, loss.replace('_',' ') ,'Normalized Dist.' , 'run_n={}'.format(params.run_n), plotname='{}plot_{}'.format(fig_dir,loss), legend=['BG']+sig_types, ylogscale=True)

# *******************************************************
#                       plotting ROCs 
# *******************************************************
print('Plotting ROCs')
for loss in losses:
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

for i_num,var in enumerate(latent_vars):
    latent_space = []
    latent_space.append(particles_dict[var]['BG'])
    latent_space.append(particles_dict[var][sig_types[0]]) #first type of signal only , for comparison
    plot.plot_features(latent_space, var ,'Normalized Dist.' , 'run_n={}'.format(params.run_n), plotname='{}plot_{}_{}'.format(fig_dir,var,'BG_SIG'), legend=('BG,{}'.format(sig_types[0])).split(','), ylogscale=True)


for i_num,var in enumerate(latent_vars):
    latent_space = []
    for sig in sig_types:
        latent_space.append(particles_dict[var][sig])
    plot.plot_features(latent_space, var ,'Normalized Dist.' , 'run_n={}'.format(params.run_n), plotname='{}plot_{}_{}'.format(fig_dir,var,'SIG'), legend=sig_types, ylogscale=True)


# *******************************************************
#                       PCA / tSNE (visualization only) 
# *******************************************************
print('PCA on latent space')

PCA_ns = [1,2]
for PCA_n in PCA_ns:
    for i_num,var in enumerate(latent_vars):
        pca = PCA(n_components=PCA_n)
        pca.fit(particles_dict[var]['BG'])
        print('{} : Explained variation per principal component in BG: {}'.format(var,pca.explained_variance_ratio_))
        latent_pca_bg_sigs = []
        for data_type in ['BG']+sig_types:
           latent_pca = pca.transform(particles_dict[var][data_type])
           latent_pca_bg_sigs.append(latent_pca)
        if PCA_n==2: plot.plot_scatter_many(latent_pca_bg_sigs, '{} PCA #0'.format(var.replace('_',' ')) ,'{} PCA #1'.format(var.replace('_',' ')) , 'run_n={}'.format(params.run_n), plotname='{}plot_PCA{}_{}_{}'.format(fig_dir,PCA_n,var.replace('_',' '),'BG_SIG'), legend=['BG']+sig_types)
        if PCA_n==1: plot.plot_hist_many(latent_pca_bg_sigs, '{} PCA #0'.format(var.replace('_',' ')) ,'Normalized Dist.' , 'run_n={}'.format(params.run_n), plotname='{}plot_hist_PCA{}_{}_{}'.format(fig_dir,PCA_n,var.replace('_',' '),'BG_SIG'), legend=['BG']+sig_types)







   
   

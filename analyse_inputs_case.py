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
import pofah.path_constants.sample_dict_file_parts_input_vae_case as sdi
#import pofah.path_constants.sample_dict_file_parts_input_vae_case_sig as sdi_sig
import pofah.util.experiment as expe
import pofah.util.sample_factory as safa
import util.data_generator_particlenet as dage_pn
import sarewt.data_reader as dare
import pofah.phase_space.cut_constants as cuts
import training as tra
import time
import h5py
import json
import util.util_plotting as plot
import time,pathlib
import matplotlib
matplotlib.use('Agg')


# ********************************************************
#       runtime params
# ********************************************************
RunParameters = namedtuple('Parameters', 'run_n  \
 epochs train_total_n gen_part_n valid_total_n batch_n learning_rate max_lr_decay lambda_reg generator')
params = RunParameters(run_n=00, 
                       epochs=40, 
                       train_total_n=int(1e4 ),  #2e6 
                       valid_total_n=int(1e3), #1e5
                       gen_part_n=int(1e5), #1e5
                       batch_n=100, 
                       learning_rate=0.001,
                       max_lr_decay=5, 
                       lambda_reg=0.0,# 'L1L2'
                       generator=0)  #run generator or not

experiment = expe.Experiment(params.run_n).setup(model_dir=True, fig_dir=True)
paths = safa.SamplePathDirFactory(sdi.path_dict)



# ********************************************************
#       prepare training (generator) and validation data
# ********************************************************
start_time = time.time()
# train (generator)
if params.generator:
    print('>>> Generator is not yet set up for case data')
else :
    # training (full tensor, 1M events -> 2M samples)
    print('>>> Preparing training dataset')
    data_train = dage_pn.prepare_prediction_data_from_dir_case(path=paths.sample_dir_path('qcdSide'),sample_part_n=params.train_total_n,shuffle=True,region='side')
    
print('Plotting consistuents features')
fig_dir = os.path.join(experiment.model_dir, 'figs/')
pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

#print(dage_pn.get_mean_and_stdev(data_train))

num_feats = data_train.shape[-1]
plot.plot_features([data_train.reshape(-1, num_feats)], 'Input Features' ,'Normalized Dist.' , 'QCD', plotname='{}plot_features_{}'.format(fig_dir,'QCD_side'), legend=['BG'], ylogscale=True)

#print('Plotting jet features')
#const,const_names,feats,feats_names = dare.CaseDataReader(paths.sample_dir_path('qcdSide')).read_const_feats_from_dir(max_n=params.train_total_n)
#plot.plot_features([feats], feats_names ,'Normalized Dist.' , 'QCD', plotname='{}plot_jet_features_{}'.format(fig_dir,'QCD_side'), legend=['BG'], ylogscale=True)

#print('Plotting jet features sig')
#paths_sig = safa.SamplePathDirFactory(sdi.path_dict)
#const,const_names,feats,feats_names = dare.CaseDataReader(paths_sig.sample_dir_path('GravToZZ_M3500')).read_const_feats_from_dir(max_n=params.train_total_n)
#plot.plot_features([feats], feats_names ,'Normalized Dist.' , 'GravToZZ_M3500', plotname='{}plot_jet_features_{}'.format(fig_dir,'GravToZZ_M3500'), legend=['GravToZZ_M3500'], ylogscale=True)
import os

import util.experiment as ex
import util.event_sample as es
import vae.losses as lo
from vae.vae_3Dloss_model import VAE_3D
from config import *

# ********************************************************
#               runtime params
# ********************************************************

run_n = 5
experiment = ex.Experiment(run_n).setup(result_dir=True)

# ********************************************
#               read test data (events)
# ********************************************

test_sample = es.EventSample.from_input_file('RS Graviton WW br 3.0TeV',os.path.join(config['input_dir'],'RSGraviton_WW_BROAD_13TeV_PU40_3.0TeV_concat_10K.h5'))
test_evts_j1, test_evts_j2 = test_sample.get_particles()

# ********************************************
#               load model
# ********************************************

vae = VAE_3D()
vae.load( experiment.model_dir )

# *******************************************************
#               predict test data
# *******************************************************

test_evts_j1_reco, z_mean_j1, z_log_var_j1 = vae.predict_with_latent(test_evts_j1)
test_evts_j2_reco, z_mean_j2, z_log_var_j2 = vae.predict_with_latent(test_evts_j2)

# *******************************************************
#               compute losses
# *******************************************************

losses_j1 = lo.compute_loss_of_prediction_3D_kl(test_evts_j1, test_evts_j1_reco, z_mean_j1, z_log_var_j1)
losses_j2 = lo.compute_loss_of_prediction_3D_kl(test_evts_j2, test_evts_j2_reco, z_mean_j2, z_log_var_j2)

# *******************************************************
#               add losses to DataSample and save
# *******************************************************

reco_sample = es.EventSample(test_sample.name + ' reco', particles=[test_evts_j1_reco,test_evts_j2_reco], event_features=test_sample.get_event_features(), particle_feature_names=test_sample.particle_feature_names)

for loss, label in zip( losses_j1, ['j1TotalLoss', 'j1RecoLoss', 'j1KlLoss']):
    reco_sample.add_event_feature(label, loss)
for loss, label in zip( losses_j2, ['j2TotalLoss', 'j2RecoLoss', 'j2KlLoss']):
    reco_sample.add_event_feature(label, loss)

# *******************************************************
#               write predicted data
# *******************************************************

reco_sample.dump(experiment.result_dir)
import sys,os
sys.path.append(os.path.abspath(os.path.join('..')))
sys.path.append(os.path.abspath(os.path.join('../sarewt_orig/')))
sys.path.append(os.path.abspath(os.path.join('../../')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setGPU
import tensorflow as tf

import pofah.util.experiment as ex
import pofah.util.event_sample as es
import vae.losses as losses
from vae.vae_particlenet import VAE_ParticleNet
import pofah.util.sample_factory as sf
import pofah.path_constants.sample_dict_file_parts_input_vae as sdi 
import pofah.path_constants.sample_dict_file_parts_input_vae_case as sdi_case
import pofah.path_constants.sample_dict_file_parts_reco_vae as sdr 
import sarewt.data_reader as dare
import pofah.phase_space.cut_constants as cuts
import training as train
import numpy as np
from sklearn.decomposition import PCA


# ********************************************************
#               runtime params
# ********************************************************
case=1
vae_type = 'vae'
latent_vars=['z_latent']
if 'vae'.lower() in vae_type:
    latent_vars+=['z_mean','z_std']
    
#test_samples = ['qcdSig', 'qcdSigExt', 'GtoWW15na', 'GtoWW15br',  'GtoWW35na', 'GtoWW35br']
test_samples ='qcdSig,qcdSideExt,GrZZ_M3p5,GrZZ_M4p5,B*TW_M3p6,WpToWZ_M3p5,WpToWZ_M4p5,Wkk_M3p5_R2,Wkk_M3p5_R8'.split(',')
#test_samples ='GrZZ_M3p5,GrZZ_M4p5,B*TW_M3p6,WpToWZ_M3p5,WpToWZ_M4p5,Wkk_M3p5_R2,Wkk_M3p5_R8'.split(',')

#test_samples ='qcdSideExt'.split(',')
#test_samples = ['qcdSideExt']

run_n = 35
cuts = cuts.sideband_cuts if 'qcdSideExt' in test_samples else cuts.signalregion_cuts #{}

experiment = ex.Experiment(run_n=run_n).setup(model_dir=True)
batch_n = 2000
n_events=int(2e6) #2e6
	
# ********************************************
#               load model
# ********************************************

vae = VAE_ParticleNet.from_saved_model(path=experiment.model_dir)
print('beta factor: ', vae.beta)
loss_fn = losses.threeD_loss


if case :
    input_paths = sf.SamplePathDirFactory(sdi_case.path_dict)
else :
    input_paths = sf.SamplePathDirFactory(sdi.path_dict)

result_paths = sf.SamplePathDirFactory(sdr.path_dict).update_base_path({'$run$': experiment.run_dir})

for sample_id in test_samples:

    # ********************************************
    #               read test data (events)
    # ********************************************


    list_ds = tf.data.Dataset.list_files(input_paths.sample_dir_path(sample_id)+'/*')

    for file_path in list_ds.take(3):

        file_name = file_path.numpy().decode('utf-8').split(os.sep)[-1]
        test_sample = es.CaseEventSample.from_input_file_normalized(sample_id, file_path.numpy().decode('utf-8'),sample_num=n_events) #, **cuts
        test_evts_j1, test_evts_j2 = test_sample.get_particles()
        test_evts_j1 = test_evts_j1
        test_evts_j2 = test_evts_j2
        print('{}: {} j1 evts, {} j2 evts'.format(file_path.numpy().decode('utf-8'), len(test_evts_j1), len(test_evts_j2)))
        test_j1_ds = tf.data.Dataset.from_tensor_slices((test_evts_j1[:,:,0:2],test_evts_j1[:,:,:])).batch(batch_n).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        test_j2_ds = tf.data.Dataset.from_tensor_slices((test_evts_j2[:,:,0:2],test_evts_j2[:,:,:])).batch(batch_n).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # *******************************************************
        #         forward pass test data -> reco and losses
        # *******************************************************

        print('predicting {}'.format(sample_id))
        reco_j1, loss_j1_reco, loss_j1_kl, latent_j1 = train.predict_particle_net(vae, loss_fn, test_j1_ds,return_latent=True)
        reco_j2, loss_j2_reco, loss_j2_kl, latent_j2 = train.predict_particle_net(vae, loss_fn, test_j2_ds,return_latent=True)
        print('prediction finished, prepared {} events'.format(test_evts_j1.shape[0]))
        losses_j1 = [losses.total_loss(loss_j1_reco.astype(np.float32), loss_j1_kl.astype(np.float32), vae.beta), loss_j1_reco, loss_j1_kl]
        losses_j2 = [losses.total_loss(loss_j2_reco.astype(np.float32), loss_j2_kl.astype(np.float32), vae.beta), loss_j2_reco, loss_j2_kl]
        # *******************************************************
        #               add losses to DataSample and save
        # *******************************************************

        reco_sample = es.CaseEventSample(sample_id + 'Reco', particles=[reco_j1, reco_j2], jet_features=test_sample.get_event_features(), particle_feature_names=test_sample.particle_feature_names)

        for loss, label in zip( losses_j1, ['j1TotalLoss', 'j1RecoLoss', 'j1KlLoss']):
            # import ipdb; ipdb.set_trace()    
            reco_sample.add_event_feature(label, loss)
        for loss, label in zip( losses_j2, ['j2TotalLoss', 'j2RecoLoss', 'j2KlLoss']):
            reco_sample.add_event_feature(label, loss)
        PCA_n = 1
        for lat,label_jet in zip([latent_j1,latent_j2],['j1','j2']):
            for z_num,label in enumerate(latent_vars):
                for dim in range(lat[z_num].shape[-1]):
                    reco_sample.add_event_feature(label_jet+label+'_'+str(dim), lat[z_num][:,dim])
                pca = PCA(n_components=PCA_n)
                latent_pca = pca.fit_transform(lat[z_num])
                print('{} : Explained variation per principal component in BG: {}'.format(label,pca.explained_variance_ratio_))
                reco_sample.add_event_feature(label_jet+label+'_pca', latent_pca)


        # *******************************************************
        #               write predicted data
        # *******************************************************
        print('writing {}'.format(sample_id))
        reco_sample.dump(os.path.join(result_paths.sample_dir_path(reco_sample.name, mkdir=True), file_name))


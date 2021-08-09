import numpy as np
import sarewt.data_reader as dare
import pofah.util.utility_fun as utfu
import h5py
import tensorflow as tf 
import sklearn.utils as skutil


def log_transform(x):
    return np.where(x==0,-10,np.log(x))

def transform_min_max(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def transform_mean_std(x):
    return (x-np.mean(x))/(3*np.std(x))


def mask_training_cuts(constituents, features):
    ''' get mask for training cuts requiring a jet-pt > 200'''
    jetPt_cut = 200.
    idx_j1Pt, idx_j2Pt = 1, 6
    idx_feat_pt = 2
    mask_j1 = features[:, idx_j1Pt] > jetPt_cut
    mask_j2 = features[:, idx_j2Pt] > jetPt_cut
    ''' normalize jet constituents pt to the jet pt'''
    for jet_idx in [0,1]:
        constituents[:,jet_idx,:,idx_feat_pt] = np.where(features[:, idx_j1Pt,None]!=0, constituents[:,jet_idx,:,idx_feat_pt]/features[:, idx_j1Pt,None],0.) 
    ''' log transform pt of constituents'''
    for jet_idx in [0,1]:
        constituents[:,jet_idx,:,idx_feat_pt] = log_transform(constituents[:,jet_idx,:,idx_feat_pt]) 
    return mask_j1, mask_j2

def constituents_to_input_samples(constituents, mask_j1, mask_j2): # -> np.ndarray
        const_j1 = constituents[:,0,:,:][mask_j1]
        const_j2 = constituents[:,1,:,:][mask_j2]
        samples = np.vstack([const_j1, const_j2])
        np.random.shuffle(samples) #this will only shuffle jets
        samples = np.array([skutil.shuffle(item) for item in samples]) #this is pretty slow though...
        return samples  

def events_to_input_samples(constituents, features):
    mask_j1, mask_j2 = mask_training_cuts(constituents, features)
    return constituents_to_input_samples(constituents, mask_j1, mask_j2)


def normalize_features(particles):
    idx_eta, idx_phi, idx_pt = range(3)
    # min-max normalize pt
    particles[:,:,idx_pt] = transform_min_max(particles[:,:,idx_pt]) 
    # standard normalize angles, then min - max to have all features in the same range, and make masking easier
    particles[:,:,idx_eta] = transform_min_max(transform_mean_std(particles[:,:,idx_eta]))
    particles[:,:,idx_phi] = transform_min_max(transform_mean_std(particles[:,:,idx_phi]))
    return particles



class DataGenerator():

    def __init__(self, path, sample_part_n=1e4, sample_max_n=None, **cuts):
        ''' 
            sample_part_n ... number of events(!) read as chunk from file-data-generator (TODO: change to event_part_n)
            sample_max_n ... number of single jet samples as input into VAE (unpacked dijets)
        '''
        self.path = path
        self.sample_part_n = int(sample_part_n) # sample_part_n events from file parts
        self.sample_max_n = int(sample_max_n) if sample_max_n else None
        self.cuts = cuts


    def __call__(self): # -> generator object yielding np.ndarray, np.ndarray
        '''
            generate single(!) data-sample (batching done in tf.Dataset)
        '''

        print('path = ',self.path)
        # create new file data-reader, each time data-generator is called (otherwise file-data-reader generation not reset)
        generator = dare.DataReader(self.path).generate_event_parts_from_dir(parts_n=self.sample_part_n, **self.cuts)

        samples_read_n = 0
        # loop through whole dataset, reading sample_part_n events at a time
        for constituents, features in generator:
            samples = events_to_input_samples(constituents, features)
            samples = normalize_features(samples)
            indices = list(range(len(samples)))
            samples_read_n += len(samples)
            while indices:
                index = indices.pop(0)
                next_sample = (samples[index,:,0:2],samples[index,:,:])
                yield next_sample
            if self.sample_max_n is not None and (samples_read_n >= self.sample_max_n):
                break
        
        print('[DataGenerator]: __call__() yielded {} samples'.format(samples_read_n))
        generator.close()


class DataGeneratorDirect():

    def __init__(self, path, sample_part_n=1e4, sample_max_n=None,batch_size=256, **cuts):
        ''' 
            sample_part_n ... number of events(!) read as chunk from file-data-generator (TODO: change to event_part_n)
            sample_max_n ... number of single jet samples as input into VAE (unpacked dijets)
        '''
        self.path = path
        self.sample_part_n = int(sample_part_n) # sample_part_n events from file parts
        self.sample_max_n = int(sample_max_n) if sample_max_n else None
        self.cuts = cuts
        self.batch_size = batch_size


    def __call__(self): # -> generator object yielding np.ndarray, np.ndarray
        '''
            generate single(!) data-sample (batching done in tf.Dataset)
        '''
        batch_size = self.batch_size
        print('path = ',self.path)
        # create new file data-reader, each time data-generator is called (otherwise file-data-reader generation not reset)
        generator = dare.DataReader(self.path).generate_event_parts_from_dir(parts_n=self.sample_part_n, **self.cuts)

        samples_read_n = 0
        # loop through whole dataset, reading sample_part_n events at a time
        for constituents, features in generator:
            samples = events_to_input_samples(constituents, features)
            samples = normalize_features(samples)


            num_to_process = self.sample_max_n if self.sample_max_n is not None else samples.shape[0]
            if samples.shape[0] < num_to_process : num_to_process = samples.shape[0]
            nb = num_to_process // batch_size
            last_batch = num_to_process % batch_size

            for ib in range(nb):
                samples_read_n += batch_size
                if (samples_read_n >= self.sample_max_n+batch_size):
                    break  
                else :  
                    yield samples[ib*batch_size:(ib+1)*batch_size,:,0:2], samples[ib*batch_size:(ib+1)*batch_size,:,:]
            '''  drop remainder batch ''' 
           # if last_batch > 0:
           #     yield samples[-last_batch:,:,0:2], samples[-last_batch:,:,:] 
            if (samples_read_n >= self.sample_max_n+batch_size):
                break        
        print('[DataGenerator]: __call__() yielded {} samples'.format(samples_read_n))
        generator.close()


class DataGeneratorDirectPrepr():

    def __init__(self, path, sample_max_n=None,batch_size=256):
     
        self.path = path
        self.sample_max_n = int(sample_max_n) if sample_max_n else None
        self.batch_size = batch_size


    def __call__(self): # -> generator object yielding np.ndarray, np.ndarray
        '''
            generate single(!) data-sample (batching done in tf.Dataset)
        '''
        batch_size = self.batch_size
        print('path = ',self.path)

        samples_read_n = 0
        with h5py.File(self.path, 'r')as inFile:
            samples = inFile['particle_bg']

            nb = samples.shape[0] // batch_size
            last_batch = samples.shape[0] % batch_size

            for ib in range(nb):
                samples_read_n += batch_size
                if self.sample_max_n is not None and (samples_read_n >= self.sample_max_n+batch_size):
                    break
                else :
                    yield samples[ib*batch_size:(ib+1)*batch_size,:,0:2], samples[ib*batch_size:(ib+1)*batch_size,:,:]
            if last_batch > 0:
                yield samples[-last_batch:,:,0:2], samples[-last_batch:,:,:]
        
        print('[DataGenerator]: __call__() yielded {} samples'.format(samples_read_n))



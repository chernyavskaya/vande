import numpy as np
import h5py
import os
from importlib import reload
import sys, os
import sarewt.data_reader as dare
import pofah.util.utility_fun as utfu



def constituents_to_input_samples(constituents, mask_j1, mask_j2): # -> np.ndarray
        const_j1 = constituents[:,0,:,:][mask_j1]
        const_j2 = constituents[:,1,:,:][mask_j2]
        samples = np.vstack([const_j1, const_j2])
        #np.random.shuffle(samples) #this will only shuffle jets
        #samples = np.array([skutil.shuffle(item) for item in samples]) #this is pretty slow though, removing it, because tensorflow will shuffle. TO DO : check that it actually does
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




class ParticleInputConversion():

    def __init__(self, path, sample_part_n=1e4, outdir='',outname='conv_input', **cuts):
        ''' 
            sample_part_n ... number of events(!) read as chunk from file-data-generator (TODO: change to event_part_n)
            sample_max_n ... number of single jet samples as input into VAE (unpacked dijets)
        '''
        self.path = path
        self.sample_part_n = int(sample_part_n) # sample_part_n events from file parts
        self.cuts = cuts
        self.outdir = outdir+'/'
        self.outname = outname


    def input_prep(self): 

        print('path = ',self.path)
        # create new file data-reader, each time data-generator is called (otherwise file-data-reader generation not reset)
        #constituents_concat, particle_feature_names, features, dijet_feature_names = dare.DataReader(self.path).read_events_from_dir(read_n=self.sample_part_n, **self.cuts)
        generator = dare.DataReader(self.path).generate_event_parts_from_dir(parts_n=self.sample_part_n, **self.cuts)

        # loop through whole dataset, saving sample_part_n events at a time
        num  = 0
        for constituents, features in generator:
            self.samples = events_to_input_samples(constituents, features)
            self.samples = normalize_features(self.samples)
            self.write_input(outname = '{}{}_{}.h5'.format(self.outdir,self.outname,num))
            num+=1


    def write_input(self,outname=''): 
        with h5py.File(outname, 'w')as outFile:
            print('Writing {} jets'.format(self.samples.shape[0]))
            print('Writing file {} '.format(outname))
            outFile.create_dataset('particle_bg', data=self.samples, compression='gzip')






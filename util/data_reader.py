import h5py
import os
import numpy as np
import glob
import pandas as pd
import operator

import sarewt.util as ut

class DataReader():
    '''
        reads events (dijet constituents & dijet features)
        from single files and directories
    '''

    def __init__(self, path):
        self.path = path
        self.jet_constituents_key = 'jetConstituentsList'
        self.jet_features_key = 'eventFeatures'
        self.dijet_feature_names = 'eventFeatureNames'
        self.constituents_feature_names = 'particleFeatureNames'
        self.constituents_shape = (2, 100, 3)
        self.features_shape = (11,)


    def get_file_list(self):
        ''' return *sorted* recursive file-list in self.path '''
        flist = []
        for path, _, _ in os.walk(self.path, followlinks=True):
            if "MAYBE_BROKEN" in path:
                continue
            flist += glob.glob(path + '/' + '*.h5')
        flist.sort()
        return flist


    def read_data_from_file(self, key, path=None):
        path = path or self.path
        with h5py.File(path,'r') as f:
            return np.asarray(f.get(key))


    def read_constituents_and_dijet_features_from_file(self, path, dtype='float32'):
        ''' returns file contents (constituents and features) as numpy arrays '''
        with h5py.File(path,'r') as f:
            features = np.asarray(f.get(self.jet_features_key), dtype=dtype)
            constituents = np.asarray(f.get(self.jet_constituents_key), dtype=dtype)
            return [constituents, features]


    def make_cuts(self, constituents, features, **cuts):
        mask = ut.get_mask_for_cuts(features, **cuts)
        constituents, features = ut.mask_arrays(constituents, features, mask=mask)
        return constituents, features


    def read_events_from_file(self, fname=None, **cuts): # -> np.ndarray, np.ndarray
        fname = fname or self.path

        try:
            constituents, features = self.read_constituents_and_dijet_features_from_file(fname) # -> np.ndarray, np.ndarray
            if cuts:
                constituents, features = self.make_cuts(constituents, features, **cuts) # -> np.ndarray, np.ndarray
        except OSError as e:
            print("\n[ERROR] Could not read file ", fname, ': ', repr(e))
        except IndexError as e:
            print("\n[ERROR] No data in file ", fname, ':', repr(e))
        except Exception as e:
            print("\nCould not read file ", fname, ': ', repr(e))

        return np.asarray(constituents), np.asarray(features)


    def extend_by_file_content(self, constituents, features, fname, **cuts):
        cc, ff = self.read_events_from_file(fname, **cuts)
        constituents.extend(cc)
        features.extend(ff)
        return constituents, features


    def append_file_content(self, constituents, features, fname, **cuts):
        cc, ff = self.read_events_from_file(fname, **cuts)
        constituents = np.append(constituents, cc, axis=0)
        features = np.append(features, ff, axis=0)
        return constituents, features


    def get_slice_of_size_stop_index(constituents, features, parts_sz_mb):
        single_event_sz = constituents[0].nbytes + features[0].nbytes
        return int(round(parts_sz_mb / single_event_sz))


    def generate_event_parts_by_size(self, flist, parts_sz_mb, **cuts):
        ''' not tested! '''

        # keep data in numpy arrays for size control
        cons_cnct = np.empty((0, *self.constituents_shape))
        feat_cnct = np.empty((0, *self.features_shape))
        samples_in_part_n = None

        for i_file, fname in enumerate(flist):
            cons_cnct, feat_cnct = self.append_file_content(cons_cnct, feat_cnct, fname, **cuts)
            sz_mb_total = (cons_cnct.nbytes + feat_cnct.nbytes) / 1024**2

            while (sz_mb_total >= parts_sz_mb): # if event sample size exceeding max size or min n, yield next chunk and reset
                if samples_in_part_n is None:
                    samples_in_part_n = self.get_slice_of_size_indices(cons_cnct, feat_cnct, parts_sz_mb)
                cons_part, feat_part = cons_cnct[-parts_n:], feat_cnct[-parts_n:] # take last parts_n samples
                cons_cnct, feat_cnct = np.resize(cons_cnct, (cons_cnct.shape[0]-parts_n, *cons_cnct.shape[1:])), np.resize(feat_cnct, (feat_cnct.shape[0]-parts_n, *feat_cnct.shape[1:]))
                yield (cons_part, feat_part)
        # if data left, yield it
        if len(feat_cnct) > 0:
            yield (cons_cnct, feat_cnct)


    def generate_event_parts_by_num(self, parts_n, flist, **cuts):
        # keeping data in lists for performance
        constituents_concat = []
        features_concat = []

        for i_file, fname in enumerate(flist):
            constituents_concat, features_concat = self.extend_by_file_content(constituents_concat, features_concat, fname, **cuts)

            while (len(features_concat) >= parts_n): # if event sample size exceeding max size or min n, yield next chunk and reset
                constituents_part, constituents_concat = constituents_concat[:parts_n], constituents_concat[parts_n:] # makes copy of *references* to ndarrays 
                features_part, features_concat = features_concat[:parts_n], features_concat[parts_n:]
                yield (np.asarray(constituents_part), np.asarray(features_part)) # makes copy of all data, s.t. yielded chunk is new(!) array (since *_part is a list) => TODO: CHECK!
        
        # if data left, yield it
        if features_concat:
            yield (np.asarray(constituents_concat), np.asarray(features_concat))


    def generate_event_parts_from_dir(self, parts_n=None, parts_sz_mb=None, **cuts):
        '''
        file parts generator
        yields events in parts_n (number of events) or parts_sz_mb (size of events) chunks
        '''
        
        # if no chunk size or chunk number given, return all events in all files of directory
        if not (parts_sz_mb or parts_n):
            return self.read_events_from_dir(**cuts)

        flist = self.get_file_list()

        if parts_n is not None:
            gen = self.generate_event_parts_by_num(int(parts_n), flist, **cuts)
        else: 
            gen = self.generate_event_parts_by_size(parts_sz_mb, flist, **cuts)

        for chunk in gen: 
            yield chunk


    def read_events_from_dir(self, read_n=None, features_to_df=False, **cuts): # -> np.ndarray, list, np.ndarray, list
        '''
        read dijet events (jet constituents & jet features) from files in directory
        :param read_n: limit number of events
        :return: concatenated jet constituents and jet feature array + corresponding particle feature names and event feature names
        '''
        print('[DataReader] read_events_from_dir(): reading {} events from {}'.format((read_n or 'all'), self.path))

        constituents_concat = []
        features_concat = []

        flist = self.get_file_list()
        n = 0

        for i_file, fname in enumerate(flist):
            constituents, features = self.read_events_from_file(fname, **cuts)
            constituents_concat.append(constituents)
            features_concat.append(features)
            n += len(features)
            if read_n is not None and (n >= read_n):
                break

        constituents_concat, features_concat = np.concatenate(constituents_concat, axis=0)[:read_n], np.concatenate(features_concat, axis=0)[:read_n]
        print('\nnum files read in dir ', self.path, ': ', i_file + 1)

        particle_feature_names, dijet_feature_names = self.read_labels_from_dir(flist)

        features = pd.DataFrame(features_concat, columns=dijet_feature_names) if features_to_df else features_concat
        return [constituents_concat, particle_feature_names, features, dijet_feature_names]


    def read_constituents_from_file(self):
        ''' return array of shape [N x 2 x 100 x 3] with
            N examples, each with 2 jets, each with 100 highest pt particles, each with features eta phi pt
        '''
        return self.read_data_from_file(self.jet_constituents_key)


    def read_constituents_from_dir(self, read_n=None): # -> np.ndarray [n x 2 x 100 x 3]
        ''' read constituents of jet 1 and jet 2 from all file parts in directory '''
        constituents, *_ = self.read_events_from_dir(read_n=read_n)
        return constituents


    def generate_constituents_parts_from_dir(self, parts_sz_mb=None, parts_n=None, **cuts):
        for (constituents, features) in self.generate_event_parts_from_dir(parts_sz_mb=parts_sz_mb, parts_n=parts_n, **cuts):
            yield constituents # -> np.ndarray parts_n or parts_sz_mb sized chunks


    def read_jet_features_from_file(self, path=None, features_to_df=False, **cuts):
        path = path or self.path
        features = self.read_data_from_file(key=self.jet_features_key, path=path)
        if cuts:
            features = features[ut.get_mask_for_cuts(features, **cuts)]
        if features_to_df:
            features = pd.DataFrame(features, columns=self.read_labels_from_file(self.dijet_feature_names))
        return features


    def read_jet_features_from_dir(self, read_n=None, features_to_df=False, **cuts):
        ''' reading only dijet feature data from directory '''
        print('[DataReader] read_jet_features_from_dir(): reading {} events from {}'.format((read_n or 'all'), self.path))

        features_concat = []
        n = 0
        flist = self.get_file_list()
        for i_file, fname in enumerate(flist):
            try:
                features = self.read_jet_features_from_file(path=fname, **cuts)
                features_concat.append(features)
                n += len(features)
            except OSError as e:
                print("\nCould not read file ", fname, ': ', repr(e))
            except IndexError as e:
                print("\nNo data in file ", fname, ':', repr(e))
            if read_n and (n >= read_n):
                break

        features_concat = np.concatenate(features_concat, axis=0)[:read_n]
        print('{} events read in {} files in dir {}'.format(features_concat.shape[0], i_file + 1, self.path))

        dijet_feature_names, = self.read_labels_from_dir(flist=flist, keylist=[self.dijet_feature_names])

        if features_to_df:
            features_concat = pd.DataFrame(features_concat, columns=dijet_feature_names)

        return [features_concat, dijet_feature_names]


    def read_labels(self, key=None, path=None):
        key = key or self.dijet_feature_names
        path = path or self.path
        return [ l.decode("utf-8") for l in self.read_data_from_file(key, path) ] # decode to unicode if (from byte str of Python2)

    def read_labels_from_file(self, fname=None, keylist=None):
        if fname is None:
            fname = self.path
        if keylist is None:
            keylist = [self.constituents_feature_names, self.dijet_feature_names]
        labels = []
        for key in keylist:
            labels.append(self.read_labels(key, fname))
        return labels

    def read_labels_from_dir(self, flist=None, keylist=None):
        if flist is None:
            flist = self.get_file_list()

        for i_file, fname in enumerate(flist):
            try:
                labels = self.read_labels_from_file(fname=fname, keylist=keylist)
                break
            except Exception as e:
                print("\nCould not read file ", fname, ': ', repr(e))
                labels = []

        return labels


    def count_files_events_in_dir(self, recursive=False, **cuts):

        features_n = 0
        files_n = 0

        flist = self.get_file_list() if recursive else glob.glob(self.path + '/*.h5')

        for i_file, fname in enumerate(flist):
            try:
                features = self.read_jet_features_from_file(path=fname, **cuts)
                features_n += len(features)
                files_n += 1
            except OSError as e:
                print("\nCould not read file ", fname, ': ', repr(e))
            except IndexError as e:
                print("\nNo data in file ", fname, ':', repr(e))
            except Exception as e:
                print("\nCould not read file ", fname, ': ', repr(e))

        return files_n, features_n

    # def __del__(self):
    #   print('[DataReader] deleting...')



class CaseDataReader(DataReader):

    # set different keys
    def __init__(self, path):
        DataReader.__init__(self, path)
        self.jet_features_key = 'jet_kinematics'
        self.dijet_feature_names_val = ['mJJ', 'DeltaEtaJJ', 'j1Pt', 'j1Eta', 'j1Phi', 'j1M', 'j2Pt', 'j2Eta', 'j2Phi', 'j2M', 'j3Pt', 'j3Eta', 'j3Phi', 'j3M']
        self.jet1_constituents_key = 'jet1_PFCands'
        self.jet2_constituents_key = 'jet2_PFCands'
        self.constituents_feature_names_val = ['Px', 'Py', 'Pz', 'E']
        self.truth_label_key = 'truth_label'

    # TODO: how to exclude "test" file? (own get_file_list function?)

    def read_jet_constituents_from_file(self, file):
        ''' return jet constituents as array of shape N x 2 x 100 x 4
            (N examples, each with 2 jets, each jet with 100 highest-pt particles, each particle with px, py, pz, E features)
        '''
        if isinstance(file, str):
            file = h5py.File(file,'r') 
        j1_constituents = np.array(file.get(self.jet1_constituents_key)) # (576902, 100, 4)
        j2_constituents = np.array(file.get(self.jet2_constituents_key)) # (576902, 100, 4)
        return np.stack([j1_constituents, j2_constituents], axis=1)


    def read_constituents_and_dijet_features_from_file(self, path):
        with h5py.File(path,'r') as f:
            features = np.array(f.get(self.jet_features_key))
            constituents = self.read_jet_constituents_from_file(f)
            return [constituents, features]

    def read_labels(self, key, path=None):
        ''' labels are not provided in CASE dataset '''
        if key == self.dijet_feature_names:
            return self.dijet_feature_names_val
        if key == self.constituents_feature_names:
            return self.constituents_feature_names_val


    def read_events_from_dir(self, max_n=1e9):
        '''
        read dijet events (jet constituents & jet features) from files in directory
        :param max_n: limit number of events
        :return: concatenated jet constituents and jet feature array + corresponding particle feature names and event feature names
        '''
        print('reading', self.path)

        constituents_concat = []
        features_concat = []
        truth_labels_concat = []

        flist = self.get_file_list()

        for i_file, fname in enumerate(flist):
            try:
                constituents, features = self.read_constituents_and_dijet_features_from_file(fname)
                truth_labels = self.read_data_from_file(self.truth_label_key, fname)
                constituents_concat.extend(constituents)
                features_concat.extend(features)
                truth_labels_concat.extend(truth_labels)
            except OSError as e:
                print("\nCould not read file ", fname, ': ', repr(e))
            except IndexError as e:
                print("\nNo data in file ", fname, ':', repr(e))
            if len(constituents_concat) > max_n:
                break

        print('\nnum files read in dir ', self.path, ': ', i_file + 1)

        return [np.asarray(constituents_concat), self.constituents_feature_names_val, np.asarray(features_concat), self.dijet_feature_names_val, np.asarray(truth_labels_concat)]

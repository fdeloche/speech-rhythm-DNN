'''Class for creating inputs from TFRecords (datasets).
'''
import tensorflow as tf
from config import Config
import os
import sys
import numpy as np
import logging

from utils import lazy_property
from data import createFeaturesDescription

#striding mode
STRIDE_MODE_AVERAGE=0
STRIDE_MODE_PICK=1


#
#   Diagram pipeline
#
#
#  
#      queue                    serialize (TFRecordDataset)
#      tfrecords                + shuffle

# +--------------------------+                 +--------------------------------------+
# |    .filenames_queue      |  +---------->   |   .serialized  .serialized_batch     |
# +--------------------------+                 +------------------+-------------------+
#                                                                 |
#                                                                 |
#                                                                 |
#                                               parsing           |                           _process_example
#                                                                 |                           formatting
#                                                                 |
#                                                                 v                           (stride)
#                                              +------------------+-------------------+
#                                              |  .unprocessed   .unprocessed_batch   | +---------------------------------------+
#                                              +------------------+-------------------+                                         v
#                                                                 |                                         +-------------------+---------------------+
#                                                                 |                                         | .processed_full  .processed_full_batch  |
#                                         _process_example        |                                         +-------------------+---------------------+
#                                         formatting              |                                                             |
#                                         data augmentation       |                                                             |
#                                         (stride)                |                                                             |
#                                                                 |                                                             |
#                                                                 v                                                             |
#                                              +------------------+-------------------+                                         |
#                                              |  .processed   .processed_batch       | <---------------------------------------+
#                                              +------------------+-------------------+
#                                                                 |
#                                                                 |                                         look-up table (weights, filenames)
#                                       _process_example2         |
#                                       + unbatch                 |
#                                       + _split_tensor           |
#                                                                 |
#                                                                 |
#                                                                 v
#                                              +------------------+-------------------+
#                                              | .sliced  .sliced_batch               |
#                                              +--------------------------------------+



class NetworkInput(object):
    '''Class handling the data pipeline and generating datasets.
    Note: NetworkInput has an attribute nbr_batchs, but it does
    not correspond to the total number of batches of the splitted
    dataset as seen by Keras.
    To compute the total number of batches, use nbr_batchs*num_slices_by_example'''

    def __init__(self, config, folder='./Scores', subfolder=None, name='', 
                for_evaluation=False, data_augmentation=False, 
                languages=None, languages_model=None,
                 fs=16000, w_size=256, step=256, stride=2, initial_sample_length=10*2**14, 
                 TFRecords_batch_size=16, n_threads_reading=None, shuffle_buffer=2000,
                 verbose=True, features_description=None, F0_binary_values=False, use_deltas=False,
                 striding_mode=STRIDE_MODE_AVERAGE):
        '''
        :param config: config class object (see config.py)
        :param languages: autodetect if None
        :param languages_model: list of languages considered by the model. If None, same as 'languages' (languages of the dataset). Must contain all the languages of the dataset.
        :param data_augmentation: distorts inputs (RMS and F0)
        :param for_evaluation: if this dataset is for evaluation only (then the look-up table for sample weights will not be computed and sample weights will be equal to 1)
        :param stride: subsampling for the generated inputs (w.r.t the sampling of TFrecords). Caution: sample length must be a mutiple of stride.
        :param subfolder: string or list of strings (e.g. ["fold_0", "fold_1"]). if None, autodetect
        :param n_threads_reading: number of threads for reading TFRecords (map). Set to tf.data.experimental.AUTOTUNE if None. Can be None.
        :param shuffle_buffer: buffer size for shuffle.
        :param features_description: A description of features (dict) for reading the TFRecords. If None, calls createFeaturesDescription in data.py to create one (default will be 3 depths RMS + HP RMS + F0).
        :param F0_binary_values: if True and F0 in features description, F0 takes only the values 0 (unvoiced) and 1 (voiced)
        :param use_deltas: if True, add deltas to the input vectors (NB: for intensity vectors, delta_dB is applied, corresponding to multiplication factors. NB2: features are in order (feature1, feature1_delta, feature2, feature2_delta, ...))
        :param striding_mode: STRIDE_MODE_AVERAGE (default) or STRIDE_MODE_PICK, take average (or not) of consequent windows when striding
        '''

        repr_st=f'\nDATASET {name} \n'
        if for_evaluation:
            repr_st+="  for evaluation only (test/validation set) \n"

        self.config=config
        self.filenames_folder = folder
        self.stride = tot_stride= stride
        self.scores_fs = fs*1./(tot_stride*step)
        self.w_size=w_size
        self.step=step
        self.sample_length= initial_sample_length-initial_sample_length%self.w_size #must be a multiple of w_size
        self.sample_length=self.sample_length//step
        try:
            assert self.sample_length%self.stride == 0
        except AssertionError as e:
            logging.warning("the initial size of one example (number of timesteps) is not a multiple of stride")
        self.sample_length = self.sample_length//tot_stride
        self.name=name
        self.striding_mode=striding_mode
        self.verbose=verbose
        self.for_evaluation=for_evaluation
        self.data_augmentation=data_augmentation
        self.use_deltas=use_deltas
        self.F0_binary_values=F0_binary_values
        if for_evaluation and data_augmentation:
            print('WARNING; data augmentation not recommended for a valid/test set')
        if self.data_augmentation:
            repr_st+='Data augmentation: on. \n'
        else:
            repr_st+='Data augmentation: off.\n'

        repr_st+="\nInput params/info: \n"
        repr_st+="   sampling frequency of inputs : {:.2f} Hz \n".format(self.scores_fs)
        repr_st+="   sample length : {} (initial sample length : {}, step : {}, stride : {}) \n".format(
                self.sample_length, initial_sample_length, step, tot_stride)
        repr_st+="   sample duration : {:.2f} s \n".format(self.sample_length*1./self.scores_fs)
        repr_st+="   batch size : {} \n".format(config.batch_size)
        repr_st+=f"   num slices by example: {self.num_slices_by_example} (num timesteps by slices: {self.config.num_steps}) \n"


        self.sample_duration = self.sample_length*1./self.scores_fs
        self.filenames = []



        if languages:
            self.languages=languages
            autodetect=False
        else:
            autodetect=True
            if self.verbose:
                repr_st+="Autodetect languages \n"
            self.languages = []
            for subdirname in os.listdir(self.filenames_folder):
            	if os.path.isdir(f'{self.filenames_folder}/{subdirname}'):
            		self.languages.append(subdirname)

        #tranform subfolder into a list :
        if subfolder is not None:
            subfolder_it = subfolder if isinstance(subfolder, list) else [subfolder]
        else:
            repr_st+="   Autodetect (sub)folders \n"
            subfolder_it=os.listdir(f"{self.filenames_folder}/{self.languages[0]}/")

        if autodetect: #autodetect languages: check that (at least 1) subfolder is available in each language, otherwise delete these languages
            nosubfolder=("" in subfolder_it) and len(subfolder_it)==1  #case where there is in fact no subfolder, not applicable
            if not(nosubfolder):
                languages2=[]
                for lang in self.languages:
                    subfolders_in_lang=False
                    list_subfolders=os.listdir(f"{self.filenames_folder}/{lang}/")              
                    for subfolder in subfolder_it:
                        #HACK #remove '/' from folder name
                        subfolder = subfolder[:-1] if subfolder[-1]=='/' else subfolder
                        if subfolder in list_subfolders:
                            languages2.append(lang)
                            break
                self.languages=languages2


        if languages_model is None:
            languages_model = self.languages
        self.languages_model = languages_model

        #test all languages in languages_model
        languages_in_languages_model = True
        for lang in self.languages:
            if not lang in languages_model:
                languages_in_languages_model=False
                break
        #assert languages_in_languages_model, "all languages in 'languages' must be in 'languages_model'."
        if not(languages in languages_model):
            print('WARNING ; some languages in the dataset are not considered by the model')
            repr_st+='WARNING ; some languages in the dataset are not considered by the model \n'

        languages_model_in_languages = True
        for lang in languages_model:
            if not lang in self.languages:
                languages_model_in_languages=False


        repr_st+=f"  languages (total: {len(self.languages)}) \n"
        languages_st_list=[f'    {k}: {lang}' for (k, lang) in enumerate(self.languages)]
        repr_st+='\n'.join(languages_st_list)
        repr_st+='\n'
        if not languages_model_in_languages:
            repr_st+="Warning: some languages considered by the model are not in the dataset \n"


        self.n_classes = len(self.languages_model)
        repr_st+=f"   (Sub)folders: {subfolder_it} \n"

        self.frequencies =np.zeros(len(self.languages))
        count = {}
        for i, l in enumerate(self.languages):
            count[l]=0
            for subfolder in subfolder_it:
                if os.path.exists("{}/{}/{}".format(self.filenames_folder, l, subfolder)):
                    listfiles = os.listdir("{}/{}/{}".format(self.filenames_folder, l, subfolder))
                    for j, filename in enumerate(listfiles):
                            filepath = "{}/{}/{}/{}".format(self.filenames_folder, l, subfolder, filename)
                            self.filenames.append(filepath)
                            #DEPRECATED
                            #for record in tf.python_io.tf_record_iterator(filepath):
                            #XXX expects all the TFRecords to have the same batch size:
                            count[l] += TFRecords_batch_size
            self.frequencies[i] = count[l]
        self.frequencies /= np.sum(self.frequencies)

        self.nbr_examples = sum([value for (key, value) in count.items()])
        self.nbr_batchs = self.nbr_examples//self.config.batch_size


        repr_st+="   Total number of examples - {} - : {} ({} batchs) \n".format(name, self.nbr_examples,
                                                                              self.nbr_batchs)
        repr_st+="   Per language : \n "
        for i, l in enumerate(self.languages):
            repr_st+="    {} : {} ({:.2f} %) \n".format(l, count[l], self.frequencies[i]*100)

        #variables useful for reading TFRecords
        self.filenames_tensor=tf.constant(self.filenames)
        self.nb_files=len(self.filenames)

        self.features_description=createFeaturesDescription() if features_description is None else features_description #for reading TFRecords
        
        if n_threads_reading is None:
            n_threads_reading = tf.data.experimental.AUTOTUNE
        self.n_threads_reading=n_threads_reading
        self.shuffle_buffer=shuffle_buffer

        #find depth of Input tensors
        input_depth0=0
        for keyName in ['rmsValue', 'HRmsValue', 'F0']:
            if keyName in self.features_description:
                input_depth0+=1

        self.input_depth=self.input_depth0=input_depth0
        if self.use_deltas:
            self.input_depth*=2

        if 'F0' in self.features_description and self.F0_binary_values:
            repr_st+='F0 takes only 2 values (0:unvoiced/1:voiced)\n'

        if self.use_deltas:
            repr_st+=f"   input depth (nb features) : {self.input_depth0} x2 (using deltas) = {self.input_depth} \n"
        else:
            repr_st+=f"   input depth (nb features) : {self.input_depth} \n"

        if self.verbose:
            print(repr_st)
        self._repr = repr_st

    def __repr__(self):
        return self._repr

    @lazy_property
    def languages_filter(self):
        '''return the filter to go from 'languages_model' to 'languages'''
        filter_=[]
        for lang in self.languages:
            filter.append(self.languages_model.index(lang))
        return filter
            

    @lazy_property
    def filenames_queue(self):
        '''Queue (Dataset) for the filenames of the TFRecords'''
        f_queue=tf.data.Dataset.from_tensor_slices(self.filenames_tensor)
        f_queue_shuffled=f_queue.shuffle(self.nb_files)
        return f_queue_shuffled

    @lazy_property
    def serialized(self):
        return self.filenames_queue.interleave(
    tf.data.TFRecordDataset, num_parallel_calls=self.n_threads_reading).shuffle(self.shuffle_buffer)

    @lazy_property
    def unprocessed(self):
        return self.serialized.map(lambda t : tf.io.parse_single_example(t, self.features_description))

    @lazy_property
    def serialized_batch(self):
        return self.serialized.batch(self.config.batch_size)

    @lazy_property
    def unprocessed_batch(self):
        return self.serialized_batch.map(lambda t : tf.io.parse_example(t, self.features_description))

    #------- PROCESSING BATCHES -----------
    #random distortion function for data augmentation
    def _random_distortion(self, t, n_sig=6, alpha_sigma=np.pi/10, a=15):
        '''
        n_sig: number of sigmoids
        alpha_sigma: angle std deviation from pi/4
        a: contraction factor (the lower the smoother)
        '''
        n=n_sig
        pts_x = tf.linspace(0.,1.,n)
        alpha_mean = np.pi/4
        alpha_min = 0.
        alpha_max = np.pi/2 - np.pi/20
        alpha = alpha_mean + alpha_sigma*tf.random.normal((n, ))
        alpha = tf.math.maximum(alpha_min, alpha)
        alpha = tf.math.minimum(alpha_max, alpha)
        delta = 1./n*tf.math.tan(alpha)

        #variability on x knots
        k=tf.range(n, dtype=tf.float32)
        xk=(2*k+1)/(2*(n-1))+1/(3*n)*tf.random.normal((n, )) 

        res=delta[0]*(-0.1+0.1*tf.random.normal((1, ))*tf.ones_like(t))
        ref=0.
        for k in range(n):
            res+=delta[k]*tf.math.sigmoid(a*(t-xk[k]))
            ref+=delta[k]*tf.math.sigmoid(a*(1-xk[k])) 
        res/=ref
        return res



    def _process_example(self, example, single=True, full=False):
        '''
        Private method for the processing of data
        Note: if for evaluation, the sample weights will all be equal to 1 (then the look-up table is not necessary to construct)
        :param single: single example (True) or batch (False)
        :param full: return all info (True) with speaker_id, etc. or only info required for training (w/ sample weights computed with a look-up table + name as index). 
        '''

        #(HACK) for use_deltas, prepare filter after roll 
        if self.use_deltas:
            if single:
                filter_shift=np.ones((self.sample_length*self.stride))
                filter_shift[0]=0
            else:
                filter_shift=np.ones((1, self.sample_length*self.stride))
                filter_shift[0,0]=0
            filter_shift=tf.constant(filter_shift, dtype=tf.float32)


        listTensors=[]
        if "rmsValue" in self.features_description:
            rmsTensor=tf.io.decode_raw(example["rmsValue"], tf.float32)

            rmsTensordB = (20.*tf.math.log(rmsTensor)/tf.math.log(10.))
            rmsTensordBInput = (rmsTensordB+100)/100.  

            if self.data_augmentation:
                rmsTensordBInput=self._random_distortion(rmsTensordBInput)

            listTensors.append(rmsTensordBInput)
            if self.use_deltas:
                rmsTensordBInput_shifted0=tf.roll(rmsTensordBInput, 1, -1)
                rmsTensordBInput_shifted=rmsTensordBInput_shifted0*filter_shift
                rmsTensordBInput_deltas=rmsTensordBInput-rmsTensordBInput_shifted
                listTensors.append(rmsTensordBInput_deltas)

        if "HRmsValue" in self.features_description:
            HRmsTensor=tf.io.decode_raw(example["HRmsValue"], tf.float32)
            #NB : 'rmsValue' assumed to be in features as well
            HRmsTensordB = 20.*tf.math.log(HRmsTensor)/tf.math.log(10.)
            HRmsTensorDiff = (HRmsTensordB - rmsTensordB)/25.   #25 max gain
            listTensors.append(HRmsTensorDiff)
            if self.use_deltas:
                HRmsTensorDiff_shifted0=tf.roll(HRmsTensorDiff, 1, -1)
                HRmsTensorDiff_shifted=HRmsTensorDiff_shifted0*filter_shift
                HRmsTensorDiff_deltas=HRmsTensorDiff-HRmsTensorDiff_shifted
                listTensors.append(HRmsTensorDiff_deltas)
        if "F0" in self.features_description:
            F0Tensor=tf.io.decode_raw(example["F0"], tf.float32)

            if self.F0_binary_values:
                F0TensorInput=tf.cast(F0Tensor>20., tf.float32)
            else:
                #NB: considers that F0 is between 0 and 700 Hz
                F0TensorInput=F0Tensor/500.

                if self.data_augmentation:
                     F0TensorInput=self._random_distortion(F0TensorInput)
            listTensors.append(F0TensorInput)


            if self.use_deltas:
                F0TensorInput_shifted0=tf.roll(F0TensorInput, 1, -1)
                F0TensorInput_shifted=F0TensorInput_shifted0*filter_shift
                F0TensorInput_deltas=F0TensorInput-F0TensorInput_shifted
                listTensors.append(F0TensorInput_deltas)
        if single:
            featuresTensor=tf.stack(listTensors, axis=1)
        else:
            featuresTensor=tf.stack(listTensors, axis=2)

        if single:
            #STRIDING (average)
            stride=self.stride
            featuresTensor_strided = featuresTensor[0::stride]
            if self.striding_mode == STRIDE_MODE_AVERAGE:
                for k in range(1, stride):
                    featuresTensor_strided+=featuresTensor[k::stride]
                    featuresTensor_strided/=stride
        else:
            stride=self.stride
            featuresTensor_strided = featuresTensor[:, 0::stride]
            if self.striding_mode == STRIDE_MODE_AVERAGE:
                for k in range(1, stride):
                    featuresTensor_strided+=featuresTensor[:, k::stride]
                    featuresTensor_strided/=stride

        language = example["language"]
        database = example["database"]
        speaker = example["speaker"]
        filename=example['filename']

        if single:
            speaker_joined= tf.strings.reduce_join([language, database, speaker], separator="_")
            classTensor = tf.dtypes.cast(tf.math.equal([language], self.languages_model), dtype=tf.float32)
        else:
            stringTensor=tf.stack([language, database, speaker], axis=1)
            speaker_joined= tf.strings.reduce_join(stringTensor, separator="_", axis=1)

        if not(full): #this test is absolutely necessary as the full processed examples are 
        #required to construct the look-up tables (avoid infinite recursive loop)
            if self.for_evaluation: #no look-up table for weights
                sample_weights=dtype=tf.ones_like(speaker_joined, dtype=tf.float32)
            else:
                sample_weights=self.weights_lookup_table.lookup(speaker_joined)
            
            if single:
                classTensor = tf.dtypes.cast(tf.math.equal(language, self.languages_model), dtype=tf.float32)
            else:
                classTensor = tf.dtypes.cast(tf.math.equal(language[:, None], self.languages_model), dtype=tf.float32)

            if not(self.for_evaluation): #no look-up table for filenames
                filename_ind=dtype=tf.ones_like(speaker_joined, dtype=tf.float32)
            else:
                filename_ind=self.filenames_lookup_table.lookup(filename)

        #return {"features": example_features_strided, "filename": filename, "dataset" : dataset,
        #        "language" : language, "output": example_output, "speaker":speaker_long}
        if full:
            return featuresTensor_strided, language, speaker_joined, database, filename
        else:
            return featuresTensor_strided, classTensor, sample_weights, filename_ind


    @lazy_property
    def processed_full(self):
        return self.unprocessed.map(lambda b:self._process_example(b, full=True))

    @lazy_property
    def processed_full_batch(self):
        return self.unprocessed_batch.map(lambda b:self._process_example(b, single=False, full=True))



    #------- LOOK UP TABLES -----------
    #------- SAMPLE WEIGHTS -----------
    @lazy_property
    def weights_dict(self, K1=20., K2=5.):
        '''Lookup table  (dict) for sample weights by speaker (offsets the dataset imbalance).
        There are two normalizations : a) 1/(K1 + n_ex) where n_ex is the number of examples for one speaker
        b) second normalization is done language by language: 1/(K2 + n_loc) where n_loc is the (soft) sum of locutors
        (n_loc is the sum of n_ex/(K1+n_ex) )
        Returns the lookup table (dict)'''
        
        if self.verbose:
            print("Computing weights for each speaker (look-up table)")
        repr_st="Global weights for each language : \n "
        def decode_string_t(st_t):
            try:
                return st_t.numpy().decode('utf-8')
            except AttributeError as e:
                print("error, numpy cast")
                print(f"tensor: {st_t}")
                sys.exit(1)
        dic={}
        for language in self.languages_model:
            dic[language] = {}

        #run epoch
        for example in self.processed_full:
            _, language, spk, _, _ = example
            language=decode_string_t(language)
            spk=decode_string_t(spk)
            if not spk in dic[language]:
                dic[language][spk] = 1
            else:
                dic[language][spk] += 1

        global_n_ex = 0
        global_sum_norm_factor = 0
        for language in self.languages_model:
            #normalization 1
            n_loc=0
            for spk in dic[language]:
                n_ex = dic[language][spk]
                dic[language][spk] = 1./(K1+n_ex)
                n_loc +=  n_ex*1./(K1+n_ex)
                global_n_ex += n_ex
            #normalization 2
            for spk in dic[language]:
                dic[language][spk] /= K2+n_loc
            global_sum_norm_factor += n_loc*1/(K2+n_loc)

            lang_weight= n_loc*1./(K2+n_loc)
            repr_st+=f"weight for language {language} : {lang_weight:.3f} \n"

        #global normalization
        for language in self.languages_model:
            #normalization 3
            for spk in dic[language]:
                dic[language][spk] /=  global_sum_norm_factor*1./global_n_ex

        #merge dictionaries
        lookup_table = {}
        for language in self.languages_model:
            lookup_table.update(dic[language])


        if self.verbose:
            print(repr_st)
        self._repr+=repr_st
        return lookup_table

    @lazy_property
    def weights_lookup_table(self):
        '''Lookup table created from the dictionary weights_dict'''
        if self.for_evaluation:
            if self.verbose:
                "dataset for evaluation, not computing weights for each speaker"
            return None
        keys, values = list(self.weights_dict.keys()), list(self.weights_dict.values())
        return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.float32), 0.)


    #------- FILENAMES -----------

    @lazy_property
    def filenames_dics(self):
        '''Lookup tables for filenames. Runs a epoch.
        Returns 2 lookup tables: int-> filename (list) and filename-> int (dict)'''
        if self.verbose:
            print("Creating 'look-up tables' for filenames")
        def decode_string_t(st_t):
            try:
                return st_t.numpy().decode('utf-8')
            except AttributeError as e:
                print("error, numpy cast")
                print(f"tensor: {st_t}")
                sys.exit(1)

        l=[]
        dic={}
        ind=0

        #run epoch
        for example in self.processed_full:
            _, _,_, _, filename_st = example
            filename=decode_string_t(filename_st)
            l.append(filename)
            dic[filename]=ind
            ind+=1


        return l, dic



    @lazy_property
    def filenames_lookup_table(self):
        '''Lookup table created from the dictionary filenames_dict  string-> index'''
        if not self.for_evaluation: #return an empty lookup table instead of runing an epoch
            return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer([''], [0.], key_dtype=tf.string, value_dtype=tf.float32), 0.)
        _, filenames_dict=self.filenames_dics   
        keys, values = list(filenames_dict.keys()), list(filenames_dict.values())
        return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.float32), 0.)


    @lazy_property
    def filenames_lookup_table2(self):
        '''Lookup table for filenames. index-> filename'''
        if not self.for_evaluation:
            return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer([0], ['no name (training)'], key_dtype=tf.int32, value_dtype=tf.string),
                'no name (training)')
        else:
            filenames_list, _ =self.filenames_dics   
            return tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(range(len(filenames_list)), 
                filenames_list, key_dtype=tf.int32, value_dtype=tf.string), 'unknown name')



    #-------

    @lazy_property
    def processed(self):
        self.weights_lookup_table         #creates the lookup table if needed, seems necessary to do it before mapping to avoid errors
        self.filenames_lookup_table
        self.filenames_lookup_table2
        return self.unprocessed.map(lambda b:self._process_example(b, full=False))

    @lazy_property
    def processed_batch(self):
        #create lookup table if needed, seems necessary to do it before mapping to avoid errors
        self.weights_lookup_table
        self.filenames_lookup_table
        self.filenames_lookup_table2
        return self.unprocessed_batch.map(lambda b:self._process_example(b, single=False, full=False))

    #------- SLICING -----------
    @lazy_property
    def num_slices_by_example(self):
        '''number of slices of size num_steps (defined in config) in one example/batch'''
        try:
            assert self.sample_length%self.config.num_steps == 0
        except AssertionError as e:
            logging.warning("the size of one example (number of timesteps) is not a multiple of num_steps")
        return self.sample_length//self.config.num_steps



    def _process_example2(self, featuresTensor, classTensor, sample_weights, filenames, single=True):
        '''2nd private method for the processing of batches, that splits the batches in smaller batches of size num_steps.
        Returns a single tensor of size (sample_length (x batch_size) x augmented_depth).
        Adds a column for signaling the beginning or end of an example (first/last slice). 1:start, -1:end.
        Adds another column for filenames as indices.
        Note: in batch mode (single:False), batch/time axes are swapped.'''

        augmented_depth=self.input_depth+self.n_classes+1+1+1  #features-output-weights-first slice of batch?-filename
        #Broadcasting
        if single:
            featuresTensor_t=featuresTensor
            broadcast_shape_a=[self.sample_length, self.n_classes]
            broadcast_shape_b=[self.sample_length, 1]
            return_shape=[self.num_slices_by_example, self.config.num_steps, augmented_depth]
            newBatch_arr=np.zeros(self.sample_length, dtype=np.float32)
            newBatch_arr[0:self.config.num_steps]=np.ones(self.config.num_steps) #first slice (begining of example)
            newBatch_arr[-self.config.num_steps:]=-np.ones(self.config.num_steps) #last slice (end of example)
        else:
            #Transpose time/batch axis
            featuresTensor_t=tf.transpose(featuresTensor, [1, 0, 2])
            broadcast_shape_a=[self.sample_length, self.config.batch_size, self.n_classes]
            broadcast_shape_b=[self.sample_length, self.config.batch_size, 1]
            return_shape=[self.num_slices_by_example, self.config.num_steps, self.config.batch_size, augmented_depth]
            newBatch_arr=np.zeros((self.sample_length, self.config.batch_size), dtype=np.float32)
            newBatch_arr[0:self.config.num_steps]=np.ones((self.config.num_steps, self.config.batch_size))#first slice (begining of example)
            newBatch_arr[-self.config.num_steps:]=-np.ones((self.config.num_steps, self.config.batch_size))#last slice (end of example)


        #Broadcasting
        newBatchTensor=tf.expand_dims(newBatch_arr, axis=-1)
        classTensor=tf.broadcast_to(classTensor, broadcast_shape_a)
        sample_weights=tf.broadcast_to(tf.expand_dims(sample_weights,-1), broadcast_shape_b)
        filenames=tf.broadcast_to(tf.expand_dims(filenames,-1), broadcast_shape_b)


        #stack
        bigFeaturesTensor=tf.concat([featuresTensor_t, classTensor, sample_weights, newBatchTensor, filenames], 
            int(not(single))+1)
        bigFeaturesTensor_reshape=tf.reshape(bigFeaturesTensor, return_shape)

        #listFeatureTensors=tf.split(bigFeaturesTensor, self.num_slices_by_example, axis=int(not(single)))

        return bigFeaturesTensor_reshape
        '''
        listSlices=[]
        newBatch=True
        for featureTensor in listFeatureTensors:
            listSlices.append(((featureTensor, newBatch), classTensor, sample_weights))
            if newBatch:
                newBatch=False
        return listSlices
        '''

    @lazy_property
    def sliced0(self):
        '''Returns a Dataset producing single tensors of size (sample_length x augmented_depth). Adds a column for signaling the beginning of an example (first slice).'''
        return self.processed.map(self._process_example2).unbatch()

    @lazy_property
    def sliced_batch0(self):
        '''Returns a Dataset producing single tensors of size (sample_length x batch_size x augmented_depth). Adds a column for signaling the beginning of an example (first slice).'''
        return self.processed_batch.map(lambda a,b,c,d:self._process_example2(a,b,c,d, single=False)).unbatch()

    def _split_tensor(self, t, single=True):
        if not(single):
            t=tf.transpose(t, [1, 0, 2]) #swap again time/batch axes
        featureTensor, classTensor, sample_weights, newExampleSignal, filenames=tf.split(t, [self.input_depth, self.n_classes, 1, 1, 1], axis=-1)

        #sample_weights, filenames and newExampleSignal must be 1D (except batch dim)
        #for filenames, also replaces by string associated with index
        if single:
            sample_weights=sample_weights[0]
        else:
            sample_weights=sample_weights[:, 0]
        sample_weights=tf.squeeze(sample_weights, axis=-1)

        if single:
            newExampleSignal=newExampleSignal[0]
        else:
            newExampleSignal=newExampleSignal[:, 0]
        #newExampleSignal=tf.squeeze(newExampleSignal, axis=-1)

        if single:
            filenames=self.filenames_lookup_table2.lookup(tf.cast(filenames[0], tf.int32))
        else:
            filenames=self.filenames_lookup_table2.lookup(tf.cast(filenames[:, 0], tf.int32))
        #filenames=tf.squeeze(filenames, axis=-1)



        return (featureTensor,  newExampleSignal, filenames), classTensor, sample_weights

    @lazy_property
    def sliced(self):
        '''Returns a Dataset producing ((featureTensor,  newExampleSignal, filenames), classTensor, sample_weights) nested tuples'''
        return self.sliced0.map(self._split_tensor)

    @lazy_property
    def sliced_batch(self):
        '''Returns a Dataset producing ((featureTensor,  newExampleSignal, filenames), classTensor, sample_weights) nested tuples'''
        return self.sliced_batch0.map(lambda t:self._split_tensor(t, single=False))





'''


    @lazy_property
    def record(self):
        return read_record(self.filename_queue, self.languages_model, self.stride, self.sample_length,
                           noise_level=self.noise_level, scores_depth=self.input_depth-1)


'''

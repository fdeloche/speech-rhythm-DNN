'''Recurrent neural network for the classification of languages (using Keras)'''

import numpy as np
import tensorflow as tf
from config import *
import input

from tensorflow.keras.layers import RNN, Input, LSTM, GRU, SimpleRNN, TimeDistributed, Dense, Softmax, Dropout

from tensorflow.keras.models import Model

from tensorflow.keras.losses import kullback_leibler_divergence, categorical_crossentropy

import logging

from functools import partial

def build_model(config, input, return_state=False, force_dropout=False):
    '''returns a Keras model of RNNs.
    NOTE: The RNNs remember states between batches. Do not forget to reset states with .reset_states()
    or use the forget_states callback defined below.
    :param config: Config object (see config.py)
    :param input: NetworkInput object (see input.py)
    :param return_state: return states for LSTM layers (see Keras documentation), only useful to retrieve cell states.
    :param force_dropout: force dropout (only on last layer). Prefer to call model with training=True'''
    main_input= Input(batch_shape=(config.batch_size, config.num_steps, input.input_depth))
    newExampleSignal=Input(shape=(1))
    filename=Input(shape=(1), dtype=tf.string)
    config=completedConfig(config)

    dropout=1-config.keep_prob
    dropout_recurrent=1-config.keep_prob_recurrent
    dropout_dense_layer=1-config.keep_prob_dense_layer

    if force_dropout:
        kwargs={'training':True}
    else:
        kwargs={}

    #Regularizers
    kernel_regularizer_l1=0
    kernel_regularizer_l2=0
    recurrent_regularizer_l1=0
    recurrent_regularizer_l2=0
    if hasattr(config, 'kernel_regularizer_l1'):
        kernel_regularizer_l1=config.kernel_regularizer_l1

    if hasattr(config, 'kernel_regularizer_l2'):
        kernel_regularizer_l2=config.kernel_regularizer_l2


    if hasattr(config, 'recurrent_regularizer_l1'):
        recurrent_regularizer_l1=config.recurrent_regularizer_l1


    if hasattr(config, 'recurrent_regularizer_l2'):
       recurrent_regularizer_l2=config.recurrent_regularizer_l2

    if any([kernel_regularizer_l1, kernel_regularizer_l2]):
        kernel_regularizer = tf.keras.regularizers.l1_l2(l1=kernel_regularizer_l1, l2=kernel_regularizer_l2)
    else:
        kernel_regularizer = None

    if any([recurrent_regularizer_l1, recurrent_regularizer_l2]):
        recurrent_regularizer = tf.keras.regularizers.l1_l2(l1=recurrent_regularizer_l1, l2=recurrent_regularizer_l2)
    else:
        recurrent_regularizer = None

    global n_layer
    n_layer=0
    def createRecurrentLayer():
        global n_layer
        if config.cell_type==LSTM_CELL:
            kernel_regularizer2 = None if n_layer==0 else kernel_regularizer #no regularization on input for first layer
            standard_dropout= 0 if n_layer==0 else dropout #no dropout of inputs for first layer 
            n_layer+=1
            return LSTM(config.hidden_size, unit_forget_bias=True, 
                kernel_regularizer=kernel_regularizer2, recurrent_regularizer=recurrent_regularizer, 
                dropout=standard_dropout, recurrent_dropout=dropout_recurrent, return_sequences=True, return_state=return_state,
                stateful=True, name=f"lstm_{n_layer}")


    h=main_input
    for i in range(config.num_layers):
        if return_state:
            h, state_h, state_c=createRecurrentLayer()(h)
        else:
            h=createRecurrentLayer()(h)
        d={}


    dropout_layer=Dropout(dropout_dense_layer, noise_shape=(config.batch_size, 1, config.hidden_size))
    h=dropout_layer(h, **kwargs) #kwargs: optionally force dropout during test

    dense_layer=TimeDistributed(Dense(input.n_classes))
    logits=dense_layer(h)
    y_=Softmax()(logits)

    return Model(inputs=[main_input, newExampleSignal, filename], outputs=y_)



# --------- METRICS ------------
# Need to define custom metrics as only the last step is considered

def verbose_metric(y_true, y_pred):
    '''only for test purpose'''
    print_op1=tf.print(y_true)
    print_op2=tf.print(y_pred)
    with tf.control_dependencies([print_op1, print_op2]):
        return 0


def KL_div_on_last_step(y_true, y_pred):
    y2_true = y_true[:, -1]
    y2_pred = y_pred[:, -1]

    return kullback_leibler_divergence(y2_true, y2_pred)


def cross_entropy_on_last_step(y_true, y_pred):
    y2_true = y_true[:, -1]
    y2_pred = y_pred[:, -1]

    return categorical_crossentropy(y2_true, y2_pred)

def accuracy_on_last_step(y_true, y_pred):
    y2_true = y_true[:, -1]
    y2_pred = y_pred[:, -1]

    y2_class_pred = tf.math.argmax(y2_pred, axis=-1)
    y2_class_true = tf.math.argmax(y2_true, axis=-1)

    return tf.cast(tf.math.equal(y2_class_pred, y2_class_true), dtype=tf.float32) #no need for reduce_mean, handled by Keras

def top_k_accuracy_on_last_step(y_true, y_pred, k=3):
    y2_true = y_true[:, -1]
    y2_pred = y_pred[:, -1]

    _, y2_topk = tf.math.top_k(y2_pred, k=k)
    y2_class_true = tf.math.argmax(y2_true, output_type=tf.dtypes.int32, axis=-1)[:, None] #expand dims

    res=tf.cast(tf.math.equal(y2_topk, y2_class_true), dtype=tf.float32)
    return tf.math.reduce_sum(res, axis=-1) #no need for reduce_mean, handled by Keras


def top_k_accuracy_on_last_step_partial(k=3):
    '''returns the right metric (function) for k'''
    res=partial(top_k_accuracy_on_last_step, k=k)
    res.__name__=f"top_{k}_accuracy_on_last_step"
    return res


#metrics using a restricted number of classes (languages)

def accuracy_on_last_step_selected(y_true, y_pred, filter):
    y2_true = y_true[filter, -1]
    y2_pred = y_pred[filter, -1]

    y2_class_pred = tf.math.argmax(y2_pred, axis=-1)
    y2_class_true = tf.math.argmax(y2_true, axis=-1)

    return tf.cast(tf.math.equal(y2_class_pred, y2_class_true), dtype=tf.float32) #no need for reduce_mean, handled by Keras

def top_k_accuracy_on_last_step_selected(y_true, y_pred, filter, k=3):
    y2_true = y_true[filter, -1]
    y2_pred = y_pred[filter, -1]

    _, y2_topk = tf.math.top_k(y2_pred, k=k)
    y2_class_true = tf.math.argmax(y2_true, output_type=tf.dtypes.int32, axis=-1)[:, None] #expand dims

    res=tf.cast(tf.math.equal(y2_topk, y2_class_true), dtype=tf.float32)
    return tf.math.reduce_sum(res, axis=-1) #no need for reduce_mean, handled by Keras



def top_k_accuracy_on_last_step_selected_partial(filter, k=3):
    '''returns the right metric (function) for k'''
    res=partial(top_k_accuracy_on_last_step_partial, filter, k=k)
    res.__name__=f"top_{k}_accuracy_on_last_step"
    return res


#Stateless metrics
#Keras averages the metrics for every batches, and not only at the end of one sequence, i.e. at the end of (non-splitted) batches.
#The metrics defined below avoid this problem by custom updates/averaging

class MyStateLessMetric(tf.keras.metrics.Metric):
    '''Parent class for stateless metrics (with accumulator). Only computes metrics at end of seq (or every num_slices_by_example splitted batches with a given initial batch index)'''
    
    def __init__(self, input_, metric_func, includeSampleWeights=True, 
        name=None, ind_batch_compute=-1, **kwargs):
        '''
        :param input_: networkInput object
        :param metric_func: functions defining metric (cf keras and stateful metrics)
        :param includeSampleWeights: metrics will be multiplied by sample weights inside each batch
        :param name: if None, will take name of function + _stateless
        :param ind_batch_compute: the metric is computed every num_slices_by_example (splitted) batches beginning 
        with index ind_batch_compute (first = 0, last=-1 default)'''
        if name is None:
            name=f"{metric_func.__name__}_stateless"
        super(MyStateLessMetric, self).__init__(name=name, **kwargs)
        self.acc = self.add_weight(name='acc', initializer='zeros')
        self.acc.assign(-1)
        self.includeSampleWeights=includeSampleWeights
        self.batch_count=self.add_weight(name='nb_batchs', initializer='zeros') #count batches
        self.num_slices_by_example=input_.num_slices_by_example
        self.metric_func=metric_func
        self.ind_batch_compute=ind_batch_compute%self.num_slices_by_example

    def update_state(self, y_true, y_pred, sample_weight=None):
        with tf.control_dependencies([self.batch_count.assign_add(1)]):
            values = self.metric_func(y_true, y_pred)

            if self.includeSampleWeights and sample_weight is not None:
                sample_weight = tf.cast(sample_weight, self.dtype)
                values = tf.multiply(values, sample_weight)
            lastBatchSeqBool = tf.equal((self.batch_count-1)%self.num_slices_by_example, self.ind_batch_compute)
            lastBatchSeqFloat = tf.cast(lastBatchSeqBool, tf.float32)
            self.acc.assign_add(lastBatchSeqFloat*tf.reduce_mean(values))

    def result(self):
        return self.acc/(self.batch_count//self.num_slices_by_example+1e-5)

    def reset_states(self):
        self.acc.assign(0)
        self.batch_count.assign(0)


class AccuracyStateless(MyStateLessMetric):
    def __init__(self, input_, includeSampleWeights=True,  
        name=None, ind_batch_compute=-1, **kwargs):
        '''
        :param input_: networkInput object
        :param includeSampleWeights: metrics will be multiplied by sample weights inside each batch
        :param name: if None, will take name 'accuracy_at_end_of_sequences' or 'accuracy_slice_k'  (k begins at 0)
        :param ind_batch_compute: the metric is computed every num_slices_by_example (splitted) batches beginning with index ind_batch_compute (first = 0, last=-1 default)'''
        if name is None:
            if ind_batch_compute%input_.num_slices_by_example==input_.num_slices_by_example-1:
                name='accuracy_at_end_of_sequences'
            else:
                name=f'accuracy_slice_{ind_batch_compute}'
        super(AccuracyStateless, self).__init__(input_, accuracy_on_last_step, includeSampleWeights=includeSampleWeights,  
            name=name, ind_batch_compute=ind_batch_compute, **kwargs)


class TopKAccuracyStateless(MyStateLessMetric):
    def __init__(self, input_, k=3, includeSampleWeights=True,  
        name=None, ind_batch_compute=-1, **kwargs):
        '''
        :param input_: networkInput object
        :param includeSampleWeights: metrics will be multiplied by sample weights inside each batch
        :param name: if None, will take name 'accuracy_at_end_of_sequences' or 'accuracy_slice_k'  (k begins at 0)
        :param ind_batch_compute: the metric is computed every num_slices_by_example (splitted) batches beginning with index ind_batch_compute (first = 0, last=-1 default)'''
        if name is None:
            if ind_batch_compute%input_.num_slices_by_example==input_.num_slices_by_example-1:
                name=f'top_{k}_accuracy_at_end_of_sequences'
            else:
                name=f'top_{k}_accuracy_slice_{ind_batch_compute}'
        metric_func=top_k_accuracy_on_last_step_partial(k=k)
        super(TopKAccuracyStateless, self).__init__(input_, metric_func, includeSampleWeights=includeSampleWeights,  
            name=name, ind_batch_compute=ind_batch_compute, **kwargs)

class KL_divStateless(MyStateLessMetric):
    def __init__(self, input_, includeSampleWeights=True,  
        name=None, ind_batch_compute=-1, **kwargs):
        '''
        :param input_: networkInput object
        :param includeSampleWeights: metrics will be multiplied by sample weights inside each batch
        :param name: if None, will take name 'KL_div_at_end_of_sequences' or 'KL_div_slice_k'  (k begins at 0)
        :param ind_batch_compute: the metric is computed every num_slices_by_example (splitted) batches beginning with index ind_batch_compute (first = 0, last=-1 default)'''
        if name is None:
            if ind_batch_compute%input_.num_slices_by_example==input_.num_slices_by_example-1:
                name='KL_div_at_end_of_sequences'
            else:
                name=f'KL_div_slice_{ind_batch_compute}'
        super(KL_divStateless, self).__init__(input_, KL_div_on_last_step, includeSampleWeights=includeSampleWeights,  
            name=name, ind_batch_compute=ind_batch_compute, **kwargs)

class crossEntropyStateless(MyStateLessMetric):
    def __init__(self, input_, includeSampleWeights=True,  
        name=None, ind_batch_compute=-1, **kwargs):
        '''
        :param input_: networkInput object
        :param includeSampleWeights: metrics will be multiplied by sample weights inside each batch
        :param name: if None, will take name 'cross_entropy_at_end_of_sequences' or 'cross_entropy_slice_k'  (k begins at 0)
        :param ind_batch_compute: the metric is computed every num_slices_by_example (splitted) batches beginning with index ind_batch_compute (first = 0, last=-1 default)'''
        if name is None:
            if ind_batch_compute%input_.num_slices_by_example==input_.num_slices_by_example-1:
                name='cross_entropy_at_end_of_sequences'
            else:
                name=f'cross_entropy_slice_{ind_batch_compute}'
        super(crossEntropyStateless, self).__init__(input_, cross_entropy_on_last_step, includeSampleWeights=includeSampleWeights,  
            name=name, ind_batch_compute=ind_batch_compute, **kwargs)

# --------- CALLBACKS ------------

class Forget_states_callback(tf.keras.callbacks.Callback):
    '''Forget the states at the begining of each new sequence (split into different batches).
    Does not use the newExampleSignal input (as only batch index and metrics/losses are sent to callbacks).
    It is based on the batch index instead.'''


    def __init__(self, input_, model, verbose=False):
        ''':param input_: NetworkInput object
        :param model: the RNN model'''
        super().__init__()
        self.num_slices_by_example=input_.num_slices_by_example
        self.model=model
        self.verbose=verbose

    def onBatchBegin(self, batch):
        verbose=self.verbose
        count= batch%self.num_slices_by_example

        if count==0:
            if verbose:
                print("detected START of sequence (callback), resetting states")
            self.model.reset_states()

        elif count==self.num_slices_by_example-1:
            if verbose:
                print("detected END of sequence (callback)")
        else:
            if verbose:
                print("neither START nor END of sequence (callback)")

    def on_train_batch_begin(self, batch, logs=None):
        self.onBatchBegin(batch)

    def on_test_batch_begin(self, batch, logs=None):
        self.onBatchBegin(batch)

    def on_predict_batch_begin(self, batch, logs=None):
        self.onBatchBegin(batch)


class Simple_progressBar_callback(tf.keras.callbacks.Callback):
    def __init__(self, input_, min_step=0.05, logs=None):
        ''':param input_: NetworkInput object
        :param min_step: the progress bar will print progress when it reaches a multiple of min_step'''
        super().__init__()
        self.nbr_batchs=input_.nbr_batchs*input_.num_slices_by_example
        self.progress=0
        self.min_step=min_step
        self.next_step=min_step

    def onBatchEnd(self, batch, logs=None):
        self.progress=batch/self.nbr_batchs
        if self.progress>self.next_step:
            print(f"{self.next_step:.0%}", end="\r")
            self.next_step+=self.min_step
    


    def on_train_batch_end(self, batch, logs=None):
        self.onBatchEnd(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self.onBatchEnd(batch, logs)

    def onEnd(self):
        pass
        #print("Done")

    def on_train_end(self,logs=None):
        self.onEnd()

    def on_test_end(self, logs=None):
        self.onEnd()


    def onBegin(self):
        print("")

    def on_epoch_begin(self, epoch, logs=None):
        self.onBegin()

    def on_test_begin(self, logs=None):
        self.onBegin()

    '''
    def on_predict_batch_end(self, batch, logs=None):
        self.onBatchEnd(batch, logs)
    '''

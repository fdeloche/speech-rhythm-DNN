import tensorflow as tf
import numpy as np

from data import createFeaturesDescription
from input import NetworkInput
from config import Config
from model import *

import tensorflow.keras.optimizers as optimizers

import matplotlib.pyplot as pl

from datetime import datetime

def test_printExample(language='French', subfolder='fold_0'):
    label=language[0:3]
    TFRlist=[f"Scores/{language}/{subfolder}/{label}_0to15.tfrecords"]
    TFRdataset=tf.data.TFRecordDataset(TFRlist)
    features_description=createFeaturesDescription()
    def parseSingleExample(exampleProto, features_description):
        return tf.io.parse_single_example(exampleProto, features_description)

    singleExample=TFRdataset.map(lambda t :parseSingleExample(t, features_description))
    #test queue
    for example in singleExample.take(1):
        print(example)
        rmsValues=tf.io.decode_raw(example["rmsValue"], tf.float32)
        print(f"language: {example['language']}, name: {example['filename']}, database: {example['database']},  speaker: {example['speaker']}")
        print("first values (RMS):"+str(rmsValues[0:10]))


def test_networkInput_queue():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", subfolder=None)
    f_queue=networkInput.filenames_queue
    for st in f_queue.take(10):
        print(st)

def test_networkInput_example():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", subfolder=None)
    for example in networkInput.unprocessed.take(1):
        #print(example)
        rmsValues=tf.io.decode_raw(example["rmsValue"], tf.float32)
        print(f"language: {example['language']}, name: {example['filename']}, database: {example['database']},  speaker: {example['speaker']}")
        print("first values (RMS):"+str(rmsValues[0:10]))

def test_networkInput_shuffle():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", subfolder=None)
    for i, example in enumerate(networkInput.unprocessed.take(10)):
        print(f"{i} :  language: {example['language']}, name: {example['filename']}, database: {example['database']},  speaker: {example['speaker']}")


def test_networkInput_batch():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", subfolder=None)
    for batch in networkInput.unprocessed_batch.take(1):
        example=batch
        for i in range(32):
            print(f"{i} :  language: {example['language'][i]}, name: {example['filename'][i]}, database: {example['database'][i]},  speaker: {example['speaker'][i]}")

def test_networkInput_stride_full():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=1, subfolder=None)
    for example in networkInput.processed_full.take(1):
        rms, language, speaker, database, filename = example
        pl.figure()
        t=np.linspace(0, networkInput.sample_duration, networkInput.sample_length)
        print(f"language: {language}, database: {database}, speaker: {speaker}, filename: {filename}")
        pl.plot(t, rms[:,0], label="RMS value (log)")
        pl.show()


def test_networkInput_batch_proc_full():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    for example in networkInput.processed_full_batch.take(1):
        rms, language, speaker, database, filename = example
        print(language)
        print(tf.shape(rms))



def test_networkInput_batch_proc():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    for example in networkInput.processed_batch.take(1):
        features, classTensor, speaker_weights = example
        print(speaker_weights)
        print(classTensor)
        print(tf.shape(features))

def test_networkInput_weights_dic():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    lookupTable=networkInput.weights_dic
    print(lookupTable)


def test_networkInput_lookup_table():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    lookupTable=networkInput.weights_lookup_table
    print(lookupTable.lookup(tf.constant('French_tatoeba_spk_3')))

def test_networkInput_num_slices():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    print(f"number of slices by example: {networkInput.num_slices_by_example}")

def test_networkInput_sliced():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    for example in networkInput.sliced.take(2):
        print(example)


def test_networkInput_batch_sliced():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    for i, example in networkInput.sliced_batch.take(networkInput.num_slices_by_example).enumerate():
        if i==0 or i==(networkInput.num_slices_by_example-1):
            print(example)

def test_buildModel():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    model=build_model(config, networkInput)
    model.summary()


def test_beginend_callback():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    model=build_model(config, networkInput)
    optim=optimizers.SGD(lr=0.01)

    myCallback=Forget_states_callback(networkInput, model, verbose=True)
    myLoss=tf.keras.losses.KLDivergence()
    model.compile(optimizer=optim, loss=myLoss, metrics=['accuracy'])

    dataset= networkInput.sliced_batch.take(2*networkInput.num_slices_by_example)
    model.evaluate(dataset, verbose=0, callbacks=[myCallback])


'''
#DEPRECATED TESTS

def test_progressBarCallback():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    model=build_model(config, networkInput)
    optim=optimizers.SGD(lr=0.01)

    myCallback=Forget_states_callback(networkInput, model, verbose=False)
    myCallback2=True_Monitoring_callback(networkInput)
    myCallback3=Simple_progressBar_callback(networkInput, min_step=0.1)

    myLoss=tf.keras.losses.KLDivergence()
    model.compile(optimizer=optim, loss=myLoss, metrics=['accuracy'])

    dataset= networkInput.sliced_batch#.take(4*networkInput.num_slices_by_example)
    model.evaluate(dataset, verbose=0, callbacks=[myCallback, myCallback2, myCallback3])


def test_verboseMetric():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    model=build_model(config, networkInput)
    optim=optimizers.SGD(lr=0.01)

    myCallback=Forget_states_callback(networkInput, model, verbose=False)
    myCallback2=True_Monitoring_callback(networkInput)
    myCallback3=Simple_progressBar_callback(networkInput, min_step=0.1)

    myLoss=tf.keras.losses.KLDivergence()
    model.compile(optimizer=optim, loss=myLoss, metrics=['accuracy', verbose_metric])

    dataset= networkInput.sliced_batch.take(1)
    model.evaluate(dataset, verbose=0, callbacks=[myCallback, myCallback2, myCallback3])


def test_monitoring():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    model=build_model(config, networkInput)
    optim=optimizers.SGD(lr=0.01)

    myCallback=Forget_states_callback(networkInput, model, verbose=False)
    myCallback2=True_Monitoring_callback(networkInput, verbose=2)
    myCallback3=Simple_progressBar_callback(networkInput, min_step=0.1)

    myMetrics=[accuracy_on_last_step, top_k_accuracy_on_last_step_partial(k=2),
                    KL_div_on_last_step,cross_entropy_on_last_step]
    myLoss=tf.keras.losses.KLDivergence()
    model.compile(optimizer=optim, loss=myLoss, metrics=myMetrics)

    dataset= networkInput.sliced_batch.take(4*networkInput.num_slices_by_example)
    model.evaluate(dataset, verbose=0, callbacks=[myCallback, myCallback2, myCallback3])


def test_summaries():
    config=Config(32, 32)
    networkInput=NetworkInput(config, name="My Input", stride=2, subfolder=None)
    model=build_model(config, networkInput)
    optim=optimizers.SGD(lr=0.01)

    today=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir=f"./logdir/tests/{today}/metrics"
    summary_writer=tf.summary.create_file_writer(logdir)
    summary_writer.set_as_default()

    myCallback=Forget_states_callback(networkInput, model, verbose=False)
    myCallback2=True_Monitoring_callback(networkInput, verbose=1, 
        write_summaries=True, write_summaries_step=2)
    myCallback3=Simple_progressBar_callback(networkInput, min_step=0.1)

    myMetrics=[accuracy_on_last_step, top_k_accuracy_on_last_step_partial(k=2),
                    KL_div_on_last_step,cross_entropy_on_last_step]
    myLoss=tf.keras.losses.KLDivergence()
    model.compile(optimizer=optim, loss=myLoss, metrics=myMetrics)

    dataset= networkInput.sliced_batch.take(4*networkInput.num_slices_by_example)
    model.evaluate(dataset, verbose=0, callbacks=[myCallback, myCallback2, myCallback3])
'''

if __name__ == '__main__':
    #test_printExample()
    #test_networkInput_lookup_table()
    #test_networkInput_batch_sliced()
    #test_buildModel()
    #test_beginend_callback()
    #test_progressBarCallback()
    
    ##DEPRECATED
    ##test_verboseMetric()
    ##test_monitoring()
    ##test_summaries()
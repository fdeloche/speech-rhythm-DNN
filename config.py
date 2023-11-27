'''Class for configuration parameters (useful for model and dataset).
'''

import tensorflow as tf
import logging

#Types
RNN_CELL = 0
LSTM_CELL = 1
GRU_CELL = 2

class Config:
    def __init__(self, batch_size, num_steps, **kwargs):
        self.batch_size = batch_size
        self.num_steps=num_steps
        self.__dict__.update(kwargs)

    def __repr__(self):
        attributes = self.__dict__
        st_l = [f'{key}: {value}' for (key, value) in attributes.items() if not key.startswith('__')]
        return '\n '.join(['config: ']+ st_l)
              


class DefaultConfig(Config):
    cell_type = LSTM_CELL
    num_layers = 2
    hidden_size= 128
    keep_prob=1 #Standard dropout inputs LSTM
    keep_prob_recurrent=1 #Recurrent dropout
    keep_prob_dense_layer=1 #Dropout 
    num_steps=32
    #rq: other possible param: kernel_regularizer_l1, recurrent_regularizer_l1, kernel_regularizer_l2, recurrent_regularizer_l2

def completedConfig(config):
    defaultConfig=DefaultConfig(config.batch_size, config.num_steps)
    try:
        cell_type = config.cell_type
    except AttributeError as e:
        logging.warning("No cell type specified in config: using LSTM")
        config.cell_type=defaultConfig.cell_type
    try:
        keep_prob = config.keep_prob
    except AttributeError as e:
        config.keep_prob=keep_prob = defaultConfig.keep_prob
        logging.warning(f"'keep_prob' not specified, set to {keep_prob:.2f}")

    try:
        keep_prob_recurrent = config.keep_prob_recurrent
    except AttributeError as e:
        config.keep_prob_recurrent=keep_prob_recurrent = defaultConfig.keep_prob_recurrent
        logging.warning(f"'keep_prob_recurrent' not specified, set to {keep_prob_recurrent:.2f}")


    try:
        keep_prob_dense = config.keep_prob_dense_layer
    except AttributeError as e:
        config.keep_prob_dense_layer=keep_prob_dense_layer = defaultConfig.keep_prob_dense_layer
        logging.warning(f"'keep_prob_dense_layer' not specified, set to {keep_prob_dense_layer:.2f}")



    try:
        num_layers = config.num_layers
    except AttributeError as e:
        config.num_layers=num_layers=defaultConfig.num_layers
        logging.warning(f"'num_layers' not specified in config, set to {num_layers}")

    try:
        hidden_size = config.hidden_size
    except AttributeError as e:
        config.hidden_size=hidden_size=defaultConfig.hidden_size
        logging.warning(f"'hidden_size' not specified in config, set to {hidden_size}")
    return config

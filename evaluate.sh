#!/bin/bash

#ex 
#2x150 bis
#papermill Evaluate\ model.ipynb evaluate_2_150.ipynb -p run_with_papermill True -p hidden_size 150  -p use_deltas True -p use_F0 True -p F0_binary_values True -p useBalancedDataSet True -p keep_prob 0.9 -p keep_prob_recurrent 0.9 -p keep_prob_dense_layer 0.8 -p weights_filename './logdir/2021-08-19_14-53-14-2_150_mult_dropout_voiced_unvoiced_bis/weights/weights_2021-08-19_14-53-14.h5' -p weights_name 'weights_2_150_voiced_unvoiced_bis' -p compute_confusion_matrix True


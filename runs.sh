#!/bin/bash
#function last_dir { cd './logdir/'; echo $(ls -td -- */ | head -n 1); cd ..; }

# ex 

#2x150 using deltas multiple dropout + voiced/unvoiced
#papermill Model\ training.ipynb training.ipynb -p use_F0 True -p F0_binary_values True -p nb_epochs 25 -p hidden_size 150 -p use_deltas True -p keep_prob 0.9 -p keep_prob_recurrent 0.9 -p keep_prob_dense_layer 0.7 -p run_with_papermill True -p name_train '2_150_multiple_dropout_voiced_unvoiced'
#mv training.ipynb ./logdir/$(last_dir)/



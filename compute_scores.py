from __future__ import division

import numpy as np
import os
import argparse
import sys
import aifc
import re
from timer import Timer
import tensorflow as tf

from clear_scores_markers import clear_markers

from utils import listFiles, LoadingBar

from data import ScoresComputer, createExample

import random 

if __name__ == '__main__':

	default_batch_size=16
	#default_input_length=2**17 #8s
	default_input_length=10*2**14 #10s
	max_nb_folds=10

	#timer
	timer = Timer()
	timer.start()

	#### PARAMS ####
	parser = argparse.ArgumentParser()
	parser.add_argument('languages', nargs='+',
						help='list of languages')

	parser.add_argument("--batch_size", "-b", help=f"batch size (examples per file), default {default_batch_size}")
	parser.add_argument("--length", "-l", help=f"length of input sounds in samples (at fs), default {default_input_length}")
	parser.add_argument("--max_files", "-m", help="max input files per language(for each folder) (default: INF)")
	parser.add_argument("--clear_markers", action='store_true', help="clear markers in the data folders before computing scores")
	parser.add_argument("--RamusData", action='store_true', help="if set, runs the code for files in folder Normalized_Data_Ramus")


	args = parser.parse_args()

	languages = args.languages
	RamusData=args.RamusData


	filesFolder = "./Files_Ramus/" if RamusData else "./Files/"
	nFilesFolder = "./Normalized_Data_Ramus/" if RamusData else "./Normalized_Data/" 
	scoresFolder = "./Scores_Ramus/" if RamusData else "./Scores/"
	if "all" in languages:
		print("autodect available languages")
		languages = []
		for dir_ in os.listdir(filesFolder):
			if(os.path.isdir("{}{}".format(filesFolder, dir_))):
				languages.append(dir_)
	print("languages : ")
	print(languages)

	if args.batch_size:
		batch_size=int(args.batch_size)
	else:
		batch_size=default_batch_size

	print("batch size : {}".format(batch_size))

	if args.length:
		input_length=sample_length=int(args.length)
	else:
		input_length=sample_length= default_input_length

	fs = 16000
	print("Inputs: duration {:.3f} s at fs = {} Hz".format(sample_length*1./fs, fs))

	if args.max_files:
		max_files=int(args.max_files)
	else:
		max_files = np.inf


	#RMS normalization
	rms_value_norm_factor=10  #rms will be between 0 and 1 (most values between 0 and 0.5), max exceptionally around 1.5
	min_max_rms_value=0.017 #below that value, we consider that the normalization failed, and we normalize by /(2*max_rms)

	#Files/folders

	databases_names = ['librivox', 'tatoeba', 'voxforge', 'WLI', 'CommonVoice']

	for language in languages:
		if not os.path.exists('./Scores/'+language):
			os.makedirs('./Scores/'+language)

	corpus = "Normalized_Data_Ramus" if RamusData else "Normalized_Data" 

	#clear markers if necessary
	if args.clear_markers:
		clear_markers(languages, nFolder=nFilesFolder)

	#### COMPUTE SCORES ####
	scoresComputer = ScoresComputer()
	input_length2=input_length-input_length%scoresComputer.w_size
	output_size=input_length2//scoresComputer.step
	print(f"output size (1 row): {output_size}")

	files_count=0
	nb_to_process = 0
	loadingBar = LoadingBar()
	loadingBar.start()

	for language in languages:
		try:
			print(language)

			# DATA MANAGEMENT (folers, ...)


			sets_folders = ['TRAIN/', 'TEST/', 'VALID/']
			subfolders = [""]
			if any([os.path.exists("./{}/{}/{}".format(corpus, language, set_folder)) for set_folder in sets_folders]):
				print("Creating TRAIN, TEST, VALID folders")
				subfolders = sets_folders

			if os.path.exists("./{}/{}/fold_0".format(corpus, language)):
				list_folds=[0]
				for k in range(1, max_nb_folds):
					if os.path.exists("./{}/{}/fold_{}".format(corpus, language,k)):
						list_folds.append(k)
				print("folds found : {}".format(list_folds))
				subfolders = ["fold_{}/".format(k) for k in list_folds]

			#Create folders if necessary
			for set_folder in subfolders:
				folder_out = scoresFolder+language+"/"+set_folder
				if not(os.path.exists(folder_out)):
					os.makedirs(folder_out)

			for subfolder in subfolders:
				files_target = listFiles('./'+corpus+'/'+language+"/"+subfolder)
				folder_out = scoresFolder+language+"/"+subfolder
				loadingBar.counter = files_count = 0 #we export files each time files_count reaches a multiple of batch_size
				files_count_tot = 0
				#find correct files_count
				for files_out in os.listdir(folder_out):
					name = files_out.split("/")[-1]
					tf_m = re.search("(\d*)to(\d*)", files_out)
					if tf_m:
						files_count_tot = max(files_count_tot, int(tf_m.group(2))+1)

				examples = []
				examples_filenames = []
				#find aifc files
				files_target_aiff = []
				nb_files = 0
				for file_str0 in files_target:
					split_name = file_str0.split("/")
					name0=split_name[-2]+"_"+split_name[-1]
					name= name0.split(".")[0]
					ext = name0.split(".")[-1]
					if ext == "aiff":
						nb_files += 1
						if not(os.path.exists(file_str0+".marker")):
							files_target_aiff.append(file_str0)
				nb_processed = nb_files - len(files_target_aiff)
				if nb_files > 0:
					print("number of files in {}/{} : {} , processed (with marker) : {} ({:.2f} %)".format(
						language, subfolder, nb_files, nb_processed, nb_processed*100./nb_files))
				nb_to_process = min(len(files_target_aiff), max_files)
				random.shuffle(files_target_aiff) #shuffle
				files_target_aiff = files_target_aiff[0:nb_to_process]
				if nb_to_process<batch_size:
					print("not sufficient data to create batch")
					continue
				loadingBar.counter_max=nb_to_process
				loadingBar.resume()
				for file_str in files_target_aiff:
					examples_filenames.append(file_str)


					split_name = file_str.split("/")
					#name=split_name[-2]+"_"+split_name[-1]
					name= split_name[-1].split(".")[0]

					#find database (in name)
					example_database = "Ramus" if RamusData else "undefined"
					for database in databases_names:
						if database in file_str:
							example_database = database

					speaker = language if RamusData else split_name[-2]
					name = "{}_{}_{}_{}".format(subfolder[:-1], example_database, speaker, name)


					#READ FILE
					myfile = aifc.open(file_str)
					strsig = myfile.readframes(myfile.getnframes())

					s = np.frombuffer(strsig, dtype='>i2')*1./np.frombuffer(b'\x7f\xff', dtype='>i2')
					myfile.close()
					#s = np.asarray(s, np.float64)

					#COMPUTE SCORE
					m=m0 = np.size(s)
					m-=m%scoresComputer.w_size #must be a multiple of w_size
					n_bins = m//scoresComputer.step
					assert (n_bins == output_size), f"file duration must correspond to length defined by user (input length: {input_length}, size of input array: {m0})"
					
					scores_dict = scoresComputer.compute_scores(s)

					#Features

					#RMS, final normalization
					rmsValues=scores_dict['rmsValue']
					HRmsValues=scores_dict['HRmsValue']

					max_rms=np.amax(rmsValues)
					coeff_norm = 1./(2*max_rms) if max_rms<min_max_rms_value else rms_value_norm_factor #custom normalization
					rmsValues*=coeff_norm
					HRmsValues*=coeff_norm

					#F0
					F0Values=scores_dict['F0']

					#create example
					example = createExample(rmsValue=rmsValues, HRmsValue=HRmsValues, F0=F0Values,
						filename=name, language=language, database=example_database, speaker=speaker)
					examples.append(example)

					files_count +=1
					loadingBar.counter=files_count_tot

					files_count_tot += 1
					if files_count%batch_size == 0:  #save batch

						tfrecord_name = folder_out+language[0:3]+"_"+str(files_count_tot-batch_size)+"to"+str(files_count_tot-1)+".tfrecords"
						with tf.io.TFRecordWriter(tfrecord_name) as writer:
							for example in examples:
								writer.write(example.SerializeToString())

						# add markers for the files processed
						for example_file_str in examples_filenames:
							with open(example_file_str+".marker", "w"):
								pass
						examples = []
						examples_short_scores = []
						examples_filenames = []

						#np.savez(file_out, samp=file_samp, t=file_t, rms=file_rms, scores=file_scores)
						#np.savez(short_file_out, samp=file_samp, t=file_t, rms=file_rms, delta=file_delta_mean, s=file_s_mean)
				loadingBar.pause()
				print("\r Done                                                  ")


			#print("wait 10 s")
			#time.sleep(10)
		except KeyboardInterrupt as e:
			print(" Keyboard Interruption \nStop language {}".format(language))
			continue

	loadingBar.stop()
	loadingBar.join()
	timer.stop()
	print("Total time : {:.3f} s".format(timer.interval))

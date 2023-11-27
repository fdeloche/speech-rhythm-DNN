'''Print info on (raw) scores, in order to find appropriate normalization factors.'''

from __future__ import division

import numpy as np
import os
import argparse
import sys
import aifc
import re
from timer import Timer

from utils import listFiles, LoadingBar

from data import ScoresComputer

import random

import matplotlib.pyplot as pl

if __name__ == '__main__':
	default_input_length=10*2**14 #10s

	#timer
	timer = Timer()
	timer.start()

	#### PARAMS ####
	parser = argparse.ArgumentParser()
	parser.add_argument('languages', nargs='+',
						help='list of languages')
	parser.add_argument("--length", "-l", help=f"length of input sounds in samples (at fs), default {default_input_length}")
	parser.add_argument("--max_files", "-m", help="max input files per language (for each folder) (default: 10)")

	args = parser.parse_args()

	languages = args.languages
	if "all" in languages:
		print("autodect available languages")
		languages = []
		for dir_ in os.listdir("./Files"):
			if(os.path.isdir("./Files/{}".format(dir_))):
				languages.append(dir_)

	print("languages : ")
	print(languages)

	if args.length:
		input_length=int(args.length)
	else:
		input_length=sample_length= default_input_length

	fs = 16000
	print("Inputs: duration {:.3f} s at fs = {} Hz".format(sample_length*1./fs, fs))

	if args.max_files:
		max_files=int(args.max_files)
	else:
		max_files = 10


	#Files/folders

	databases_names = ['librivox', 'tatoeba', 'voxforge', 'WLI', 'CommonVoice']

	corpus = 'Normalized_Data'

	#### COMPUTE SCORES ####
	scoresComputer = ScoresComputer()
	input_length2=input_length-input_length%scoresComputer.w_size
	output_size=input_length2//scoresComputer.step
	print(f"output size (1 row): {output_size}")

	files_count=0
	nb_to_process = 0
	loadingBar = LoadingBar()
	loadingBar.start()



	max_rms_values = []
	max_max_rms = 0
	max_max_rms_filename=""

	min_max_rms = np.Inf
	min_max_rms_filename=""
	

	for language in languages:
		try:
			print(language)

			# DATA MANAGEMENT (folers, ...)

			sets_folders = ['TRAIN/', 'TEST/', 'VALID/']
			subfolders = [""]
			if any([os.path.exists("./{}/{}/{}".format(corpus, language, set_folder)) for set_folder in sets_folders]):
				subfolders = sets_folders

			if os.path.exists("./{}/{}/fold_0".format(corpus, language)):
				k=1
				while os.path.exists("./{}/{}/fold_{}".format(corpus, language,k)):
					k+=1
				print("{} folds found".format(k))
				subfolders = ["fold_{}/".format(i) for i in range(k)]

			for subfolder in subfolders:
				files_target = listFiles('./'+corpus+'/'+language+"/"+subfolder)
				folder_out = "./Scores/"+language+"/"+subfolder
				loadingBar.counter = files_count = 0 #we export files each time files_count reaches a multiple of batch_size
				files_count_tot = 0
				examples = []
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
						files_target_aiff.append(file_str0)
				nb_processed = nb_files - len(files_target_aiff)
				if nb_files > 0:
					print("number of files in {}/{} : {}".format(
						language, subfolder, nb_files))
				nb_to_process = min(len(files_target_aiff), max_files)
				random.shuffle(files_target_aiff) #shuffle
				files_target_aiff = files_target_aiff[0:nb_to_process]
				loadingBar.counter_max=nb_to_process
				loadingBar.resume()
				for file_str in files_target_aiff:
					split_name = file_str.split("/")
					#name=split_name[-2]+"_"+split_name[-1]
					name= split_name[-1].split(".")[0]


					#READ FILE
					myfile = aifc.open(file_str)
					strsig = myfile.readframes(myfile.getnframes())

					s = np.frombuffer(strsig, dtype='>i2')*1./np.frombuffer(b'\x7f\xff', dtype='>i2')
					myfile.close()


					#COMPUTE SCORE
					m=m0 = np.size(s)
					m-=m%scoresComputer.w_size #must be a multiple of w_size
					n_bins = m//scoresComputer.step

					assert (n_bins == output_size), f"file duration must correspond to length defined by user (input length: {input_length}, size of input array: {m0})"

					scores_dict = scoresComputer.compute_scores(s)

					#example = createExample(rmsValue=scores_dict['rmsValue'], filename=name, language=language, database=example_database, speaker=speaker)

					max_rms=np.amax(scores_dict['rmsValue'])

					if max_rms>max_max_rms:
						max_max_rms=max_rms
						max_max_rms_filename=f'{language}/{subfolder}/{file_str}'

					if max_rms<min_max_rms:
						min_max_rms=max_rms
						min_max_rms_filename=f'{language}/{subfolder}/{file_str}'

					max_rms_values.append(max_rms)

					files_count +=1
					loadingBar.counter=files_count_tot

					files_count_tot += 1

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
	print("Total time : {:.3f} s".format(timer.interval))\

	print(f"file with max rms :{max_max_rms_filename}")
	print(f"file with min rms :{min_max_rms_filename}")


	#plot histogram
	pl.figure()
	pl.title("Max rms value (by file)")
	pl.hist(max_rms_values, bins=30)
	pl.show()
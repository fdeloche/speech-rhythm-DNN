import os
import shutil
from argparse import ArgumentParser
import sys
import subprocess

parser = ArgumentParser()
parser.add_argument('languages', nargs='+',
					help='list of languages')

parser.add_argument('--folds', help="number of folds, if not mentioned it uses the train/test structure")
parser.add_argument("--folder", help="subfolder to run the code")

args = parser.parse_args()

languages = args.languages
use_folds = args.folds is not None

if use_folds:
	nbr_folds = int(args.folds)

if "all" in languages:
    languages = ["Danish", "Dutch", "English", "Finnish",
    "French", "German", "Hungarian", "Italian",
    "Japanese", "Korean", "Mandarin", "Polish",
    "Portuguese", "Russian", "Spanish",
    "Swedish", "Turkish", "Estonian", "Arabic", "Czech", "Romanian",
    "Basque", "Catalan"]

if args.folder is not None:
	os.chdir(args.folder)

#CREATE LIBRIVOX/TATOEBA... FOLDERS
root_data_folder='/home/fdeloche/Desktop/Projects/SignalProcessing/Files/'

subfolders_data=["/Examples_10", "/examples_10"] #XXX examples 10s
if use_folds:

	subfolders=["fold_{}".format(k) for k in range(nbr_folds)]

	origin_databases ={
		'tatoeba' : f'{root_data_folder}/Tatoeba/DataWithFolds/',
		'librivox' : f'{root_data_folder}/Librivox/DataWithFolds/',
		'voxforge' : f'{root_data_folder}/VoxForge/DataWithFolds/',
		'WLI' : f'{root_data_folder}/WideLanguageIndex/DataWithFolds/',
		'CommonVoice' : f'{root_data_folder}/CommonVoice/DataWithFolds/',
	}

else:
	subfolders=["TRAIN", "VALID", "TEST"]

	origin_databases ={
		'tatoeba' : f'{root_data_folder}/Files/Tatoeba/Data2/',
		'librivox' : f'{root_data_folder}/Files/Librivox/Data/',
		'voxforge' : f'{root_data_folder}/VoxForge/Data2/',
		'WLI' : f'{root_data_folder}/Files/WideLanguageIndex/Data/',
		'CommonVoice' : f'{root_data_folder}/CommonVoice/Data2/',
	}


#CREATE FOLDERS, TRAIN/TEST FOLDERS
for language in languages:
	if not(os.path.exists(language)):
		os.makedirs(language)

	os.chdir(language)
	try:
		for folder in subfolders:
			os.makedirs(folder)
	finally:
		os.chdir("..")

for language in languages:
	print(language)
	#os.chdir(language)
	try:
		for database, data_path in origin_databases.items():
			if os.path.exists("{}/{}".format(data_path, language)):
				for subfolder in subfolders:
					for subfolder_data in subfolders_data:
						#os.chdir(subfolder)
						try:
							if os.path.exists("{}/{}{}/{}".format(data_path, language, subfolder_data, subfolder)):
								print("{} Data folder exists for : {} at {}/{}{}/{}".format(
									subfolder, database, data_path, language, subfolder_data, subfolder))
								new_path=f"./{language}/{subfolder}/{database}/"
								if not(os.path.exists(new_path)):
									os.makedirs(new_path)
									#copy
									#subprocess.call(["cp", '-Lr' ,'{}/{}{}/{}'.format(database, language, subfolder_data, subfolder), './{}{}'.format(subfolder, database)]) # L is to deal with sym links.
									#create sym links instead
									root='{}/{}{}/{}'.format(data_path, language, subfolder_data, subfolder)
									for root, reader_dirs, _ in os.walk(root):
										break 
									for reader_dir in reader_dirs:
										subprocess.call(["ln", '-s' ,'{}/{}'.format(root, reader_dir), '{}/{}'.format(new_path, reader_dir)])
								else:
									print(f"Warning: folder {new_path} already exists")						
						finally:
							pass
							#os.chdir("..")
	finally:
		pass
		#os.chdir("..")


if args.folder is not None:
	os.chdir("..")

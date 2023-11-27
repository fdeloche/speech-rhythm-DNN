import os
import subprocess

from argparse import ArgumentParser
from random import sample

parser = ArgumentParser()
parser.add_argument('--languages', '-l', nargs='+',
					help='list of languages (all: automatic detection)')

parser.add_argument('--folds', '-f', nargs='+',
					help='list of fold numbers to consider')
parser.add_argument('--nb_batchs', '-n', help='number of batches by languages')


args = parser.parse_args()
nb_batchs = int(args.nb_batchs)
languages = args.languages


if "all" in languages:
	languages = ["Danish", "Dutch", "English", "Finnish",
    "French", "German", "Hungarian", "Italian",
    "Japanese", "Korean", "Mandarin", "Polish",
    "Portuguese", "Russian", "Spanish",
    "Swedish", "Turkish", "Estonian", "Arabic", "Czech", "Romanian",
    "Basque", "Catalan"]

languages_det=[]
for language in languages:
	if os.path.exists(language):
		languages_det.append(language)

print('languages detected : ' + str(languages_det))

languages=languages_det

subfolders = [f'fold_{k}' for k in args.folds]

newfoldername0=f'balanced_{nb_batchs}'
v=0
newfoldername=f'balanced_{nb_batchs}'
existfold_bool=True
while existfold_bool:
	v+=1
	newfoldername=f'{newfoldername0}_{v}'
	existfold_bool=False
	for lang in languages:
		if os.path.exists(f'{lang}/{newfoldername}'):
			existfold_bool=True
			break

print(f"Creating new subfolders {newfoldername}")


for lang in languages:
	filenames=[]
	os.mkdir(f'{lang}/{newfoldername}')
	for subfolder in subfolders:
		if os.path.exists(f'{lang}/{subfolder}'):
			new_filenames=[f'{lang}/{subfolder}/{filename}' for filename in os.listdir(f'{lang}/{subfolder}')]
			filenames+=new_filenames
	try:
		filenames = sample(filenames, nb_batchs)
	except ValueError as e:
		print(f'WARNING: not enough batchs for {lang} (only {len(filenames)})')
		print(e)
	for filename in filenames:
		name=filename.split('/')[-1]
		subfolder=filename.split('/')[-2]
		subprocess.call(["ln", '-sr' ,filename, '{}/{}/{}_{}'.format(lang, newfoldername, subfolder, name)])






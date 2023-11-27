import numpy as np
import sys

#import timit as tm #personal module
import os
import random

#import scikits.audiolab as au #deprecated
import soundfile as sf

from timer import Timer
import argparse



#### Histograms for normalization (based on TIMIT) ####
w_size = 512 # for histograms of RMS

verbose=False

def init_histogram(histo_file_exists, histo_file=None, w_size=w_size):
    '''init histogram for normalization from file or TIMIT, return (hist, bins).'''
    if not(histo_file_exists):
        print("Err: Histogram file does not exist.")
        sys.exit(1)
    else:
        print("Loading hist file from {}".format(histo_file))
        hist_dict = np.load(histo_file)
        hist = hist_dict["hist"]
        bins = hist_dict["bins"]
    return (hist, bins)

def find_normalization_factor(sig, hist, bins, w_size=w_size):
    '''find amplitude normalization factor A from signal and reference histogram and bins. Fist version was with dB values'''
    sig = sig[0:np.size(sig)//w_size*w_size]
    sig_rs = np.reshape(sig, (-1, w_size))
    sig_avg=np.mean(sig_rs, axis=1, keepdims=True)
    rms_2 = np.sqrt(np.mean((sig_rs-sig_avg)**2, axis=1)) #RMS


    #ignore 60% lower
    rms_2=np.sort(rms_2)
    n_values=len(rms_2)
    rms_2=rms_2[int(n_values*0.6):]

    #rms_concat_db_2 = 20*np.log10(rms_2+0.000001)
    m2 = 40
    corr = np.linspace(-20, 30, m2) #-20 dB to 30 dB amplification
    corr = np.power(10, corr/20.)
    min_d = np.inf
    min_corr = np.NINF
    for cor in corr:
        rms_temp=cor*rms_2
        rms_temp = np.minimum(np.maximum(rms_temp, 0.15), 0.02)
        (hist_temp, bins_temp) = np.histogram(rms_temp, bins=bins)
        hist_temp = hist_temp*1./np.size(rms_temp)
        d_hell = np.sum((np.sqrt(hist_temp)-np.sqrt(hist))**2) #Hellinger distance
        if d_hell < min_d:
            min_d = d_hell
            min_corr = cor
    if (verbose):
        print("min corr with Hellinger distance : {:.2f} dB".format(min_corr))
    extra_factor=2
    return min_corr*extra_factor

def create_sound(s, path):
    '''formt = au.Format('aiff')
    f = au.Sndfile(path+'.aiff', 'w', formt, 1, 16000)
    f.write_frames(s
    f.close()
    '''
    sf.write(f"{path}.aiff", s, 16000)

if __name__ == '__main__':
    print('Normalization of sound files')
    timer = Timer()
    timer.start()

    #### PARAMS #####
    parser = argparse.ArgumentParser()

    parser.add_argument('languages', nargs='+',
                        help='list of languages')

    parser.add_argument("--max_files", "-m", help="max input files per language(for each folder)")

    parser.add_argument("--RamusData", action='store_true', help="if set, runs the code for files in folder Files_Ramus")

    parser.add_argument("--histogram_file", "-f", help="path for histogram file (.npz). Default: normalization_histogram.npz")

    args = parser.parse_args()

    languages = args.languages
    RamusData=args.RamusData

    if RamusData:
        print("Running code for files in /Files_Ramus")

    filesFolder = "./Files_Ramus/" if RamusData else "./Files/"
    nFilesFolder = "./Normalized_Data_Ramus/" if RamusData else "./Normalized_Data/" 
    if "all" in languages:
        print("autodect available languages")
        languages = []
        list_dir = os.listdir(filesFolder)

        subfolder=""
        '''
        if "TRAIN" in list_dir:
            print("search in folder TRAIN")
            subfolder = "TRAIN/"
            list_dir = os.listdir("./Files/TRAIN")
        '''
        for dir_ in list_dir:
            if(os.path.isdir("{}{}{}".format(filesFolder, subfolder, dir_))):
                languages.append(dir_)


    print("languages : ")
    print(languages)

    databases_names = ['librivox', 'tatoeba', 'voxforge', 'WLI', 'CommonVoice']
    if args.max_files:
        max_files=int(args.max_files)
    else:
        max_files = np.inf


    if args.histogram_file:
        histo_file=args.histogram_file
    else:
        histo_file = "normalization_histogram/normalization_histogram.npz"

    histo_file_exists = os.path.exists(histo_file)

    (hist, bins) = init_histogram(histo_file_exists, histo_file=histo_file)

    def listFiles(str_dir, pick=-1): #LIST WAVE FILES
        l = []
        for root, subFolders, files in os.walk(str_dir, followlinks=True):
            for f in files:
                #if WAV
                if (f.split('.')[-1]=='wav' or f.split('.')[-1]=='WAV'):
                    l.append(os.path.join(root, f))
        #print("number of files : " + str(len(l)))
        if pick==-1 or pick==np.inf:
            return l
        else:
            pick = min(len(l), pick)
            return random.sample(l, pick)



    for language in languages:
        if not os.path.exists(filesFolder+language):
            os.makedirs(filesFolder+language)

        if not os.path.exists(nFilesFolder+language):
            os.makedirs(nFilesFolder+language)

    for language in languages:
        print(language)

        #TEST IF FOLDERS 'TRAIN', 'TEST', 'VALID' EXIST
        if os.path.exists(filesFolder+language+'/TRAIN') or os.path.exists(filesFolder+language+'/TEST') or os.path.exists('./Files/'+language+'/VALID'):
            print("Creating TRAIN, TEST, VALID folders")
            subfolder_list = ["TRAIN/", "TEST/", "VALID/"]
            for subfolder in subfolder_list:
                if not os.path.exists(nFilesFolder+language+"/"+subfolder):
                    os.makedirs(nFilesFolder+language+"/"+subfolder)
        else:
            #try to see if the folder contains folds
            fold_id=0
            subfolder_list=[]
            while os.path.exists('{}/{}/fold_{}'.format(filesFolder, language, fold_id)):
                subfolder_list.append("fold_"+str(fold_id))
                fold_id+=1
            if fold_id==0: #no fold was found
                subfolder_list = [""] #Note: RamusData -> no subfolder

        for subfolder in subfolder_list:
            list_files = listFiles(filesFolder+language+'/'+subfolder, pick=max_files)
            print("{} number of files : {}".format(subfolder, len(list_files)))

            for str_file in list_files:
                name = os.path.splitext(os.path.basename(str_file))[0]
                #HACK add database name to file name
                for database_name in databases_names:
                    if(database_name in str_file):
                        name = "{}_{}".format(database_name, name)
                        break
                if RamusData:
                    name="Ramus_{}".format(name)
                if (verbose):
                    print("File : {}".format(name))
                sig, fs=sf.read(str_file)
                assert fs == 16000
                sig-=np.mean(sig)
                A = find_normalization_factor(sig, hist, bins)
                s_2 = A*sig

                #HACK Ramus Data, pad array
                if RamusData:
                    s_2=np.pad(s_2, ((2**14-16000)*3, 0) )

                output_str_file = str_file.replace(filesFolder, nFilesFolder)
                output_str_file = output_str_file.replace(".wav", "") #remove extension

                if not(os.path.exists(os.path.dirname(output_str_file))):
                    os.makedirs(os.path.dirname(output_str_file)) #make dirs
                create_sound(s_2, output_str_file)


    timer.stop()
    print("Total time : {:.3f} s".format(timer.interval))

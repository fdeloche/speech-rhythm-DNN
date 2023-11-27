# -*- coding: utf-8 -*-
"""
Created on Fri 30 June 2016

@author: francois


Some functions to facilitate the use of TIMIT
"""

import os
import random

import numpy as np
import matplotlib.pyplot as pl

import soundfile as sf

Stops = ['b', 'd', 'g', 'p', 't', 'k', 'dx', 'q']
Affricates = Affricatives = ['jh', 'ch']
Fricatives = ['s', 'sh', 'z', 'zh', 'f',
              'th', 'v', 'dh']
Nasals=['n', 'm', 'ng',
               'em', 'en', 'eng', 'nx']
SemiVowels = Glides = ['l', 'r', 'w', 'y', 'hh', 'hv', 'el']
Vowels = ['iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow',
          'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ax-h']
Others =['pau', 'epi', 'h#', '1', '2']

Phonemes = Stops + Affricatives + Fricatives + Nasals+ SemiVowels + Vowels + Others

Affricates_rl = ['jh_rl', 'ch_rl']

Affricates_rl2 = ['jh_rl2', 'ch_rl2'] #second part of release

Stops_rl = ['b_rl', 'd_rl', 'g_rl', 'p_rl', 't_rl', 'k_rl']
Stops_rl2 = ['b_rl2', 'd_rl2', 'g_rl2', 'p_rl2', 't_rl2', 'k_rl2'] #second part of release
Phonemes_rl = Phonemes + Affricates_rl + Stops_rl  + Affricates_rl2 + Stops_rl2  #Phonemes and release parts for stops and affricates ('_rl' and '_rl2')

def add_rl_to_cat(cat):
    '''add rl and rl2 to a category (e.g fricatives)'''
    for phn in cat[:]:
    	if not "_rl" in phn:
    		if not "{}_rl".format(phn) in cat:
    			cat.append("{}_rl".format(phn))
    		if not "{}_rl2".format(phn) in cat:
    			cat.append("{}_rl2".format(phn))

def add_rl_to_phn(phn):
    '''return a list with rl and rl2 added. Return [phn] without addition if already a release part.'''
    if not "_rl" in phn:
        return [phn, "{}_rl".format(phn), "{}_rl2".format(phn)]
    else:
        return [phn]



def listTimitFiles(str_dir, pick=-1):
    '''List all the Timit files recursively in a directory
    Pick : number of files to pick'''
    l = []
    for root, subFolders, files in os.walk(str_dir):
        for f in files:
            if '.phn' in f or '.PHN' in f:
                name = f.split('.')[0]
                l.append(os.path.join(root, name))
    print("number of files : " + str(len(l)))
    if pick==-1:
        return l
    else:
        pick = min(len(l), pick)
        return random.sample(l, pick)



'''Class for one specific Timit file'''
class TimitFile:

    def __init__(self, str_file):
        '''str_file : path to the file (without extension) '''
        self.str_file = str_file

    def parse(self, rl_mode=False, rl2_mode=True, cl_mode=True):
        '''return in a list the intervals and the corresponding class of phoneme [begin, end, phn]
        :param rl_mode: if True also consider releases for stops and affricates
        :param rl2_mode: if True splits releases into first part (rl) and second part (rl2). rl_mode has to be set to True.
        :param cl_mode: if True, the whole phn is also included (closure is always ignored). rl_mode has to be set to True to have an effect.'''
        input_file = self.str_file +".PHN"
        res = []
        res_rl = []
        closure = False
        with open(input_file) as inputfile:
            for line in inputfile:
                temp = line.strip().split(' ')
                phn = temp[2]
                if not(closure):
                    begin = int(temp[0])
                if 'cl' in phn or 'CL' in phn:
                    closure = True
                else:
                    res.append((begin, int(temp[1]), phn))

                    if closure:
                        if cl_mode: #(whole utterance)
                            res_rl.append((int(temp[0]), int(temp[1]), phn))
                        if rl2_mode:
                            begin = int(temp[0])
                            end=int(temp[1])
                            res_rl.append((begin, (begin+end)/2, phn+"_rl"))
                            res_rl.append(((begin+end)/2, end, phn+"_rl2"))
                        else:
                            res_rl.append((int(temp[0]), int(temp[1]), phn+"_rl"))
                    else:
                        res_rl.append((begin, int(temp[1]), phn))

                    closure = False
        self.list_phn = res
        self.list_phn_rl = res_rl
        if rl_mode:
            return res_rl
        else:
            return res

    def extract_signal(self):
        '''You must extract signal before doing anything else. Return false if the file does not exist'''
        exist = os.path.exists(self.str_file+".WAV")
        if (exist):
            #f = au.Sndfile(self.str_file+".WAV", 'r') #DEPRECATED
            data, samplerate = sf.read(self.str_file+".WAV")
            self.n = np.size(data)
            #TIMIT files are mono
            self.signal = data
        return exist
        #return self.signal

    def get_signal(self):
        return self.signal
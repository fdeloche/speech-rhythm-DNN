#Récupération liste des fichiers transcrits

import os
os.chdir("/Users/violette_dau/stage/lbn")
files=os.listdir("/Users/violette_dau/stage/lbn")
files.remove("ESP1167.LBN")
files.remove(".DS_Store")

Lfiles=[]

for f in files:
    Lfiles.append(f.split('.')[0])
    
##Récupération path of audio files

os.chdir("/Users/violette_dau/stage")

l=[]
Lfixed=[]
for root, dirs, files in os.walk('corpus'):
    for f in files:
        if '.wav' in f:
            name=f.split('.')[0]
            if name.upper() in Lfiles:
                l.append(os.path.join(root,name))
            if "_fixed" in name and name.split('_')[0].upper() in Lfiles:
                l.remove(os.path.join(root,name.split('_')[0]))
                l.append(os.path.join(root,name))

##Normalisation
import wave
import soundfile as sf
import numpy as np

filestopad=[]
filestocut=[]

for file in l:
    obj=wave.open(file+".wav","rb")
    len=obj.getnframes()
    if len<48000:
        filestopad.append([file,len])
    elif len>48000:
        filestocut.append([file,len])
    obj.close()

for file in filestopad:
    tps=48000-file[1]
    data=np.zeros(tps,'float64')
    obj=sf.SoundFile(file[0]+".wav",'r+')
    obj.seek(0,sf.SEEK_END)
    obj.write(data)
    obj.close()

for file in filestocut:
    obj=sf.SoundFile(file[0]+".wav",'r+')
    obj.truncate(48000)
    obj.close()

for file in l:
    obj=wave.open(file+".wav","rb")
    len=obj.getnframes()
    if len!=48000:
        print(file)
    obj.close()

#Figures comparatives en fonction des 2 meilleures cellules et par langue, avec les 160 fichiers de la base de F. Ramus (en rouge) et tous les autres fichiers (en bleu et noir)

import os
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import math



os.chdir("../activations/Scores_Ramus/weights_2020-04-07-b-1")
#os.chdir("../activations/Scores_Ramus/weights0904-2d-30Hz-newinputs-dataaugmentation-30epochs")
files=os.listdir("./")


d_act={}
for file in files:
    with open(file,'r') as json_file:
        data=json.load(json_file)
    d_act[file]=data
    
#cellule en abscisse
xlstm='lstm_2'
xcelltype='cell_states'
xcellnumber='82'
xlabel=xlstm+" "+xcelltype+" "+xcellnumber

#cellule en ordonnée
ylstm='lstm_2'
ycelltype='cell_states'
ycellnumber='38'
ylabel=ylstm+" "+ycelltype+" "+ycellnumber

##
os.chdir("../../..")
os.chdir('./compar')

#cellule en abscisse (fichier Ramus)
Daudiox={}
for langue in D:
    Daudiox[langue]={}
    for filename in D[langue]:
        if filename in d_match:
            fileaudio=d_match[filename]
            activ=datay[fileaudio.split('.')[0]][xlstm][xcelltype][int(xcellnumber)]
            Daudiox[langue][fileaudio]=math.sqrt(float(activ)**2)

#cellule en ordonnée (fichiers Ramus)
Daudioy={}
for langue in D:
    Daudioy[langue]={}
    for filename in D[langue]:
        if filename in d_match:
            fileaudio=d_match[filename]
            activ=datay[fileaudio.split('.')[0]][ylstm][ycelltype][int(ycellnumber)]
            Daudioy[langue][fileaudio]=math.sqrt(float(activ)**2)

#Figure
for langue in D: #en rouge les points correspondants aux fichiers de la base de F.Ramus
    std=np.std(list(Daudiox[langue].values()))
    xsterr=std/math.sqrt(len(list(Daudiox[langue].values())))
    x=np.mean(list(Daudiox[langue].values()))
    std=np.std(list(Daudioy[langue].values()))
    ysterr=std/math.sqrt(len(list(Daudioy[langue].values())))
    y=np.mean(list(Daudioy[langue].values()))
    plt.errorbar(x,y,label=langue,fmt=".r",xerr=xsterr,yerr=ysterr,capsize=2)
    plt.text(x+0.0005,y+0.0005,langue)


langues={'English':'en','French':'fr','Polish':'pol','Japanese':'ja','Catalan':'cat','Spanish':'esp','Dutch':'du','Italian':'it'}



#cellule en abscisse
dlanguesx={}

for file in list(d_act.keys()):
    language=d_act[file]['label']
    if language not in dlanguesx:
        dlanguesx[language]=[]
    activ=float(d_act[file]['activations'][xlstm][xcelltype][xcellnumber])
    dlanguesx[language].append(activ)
 
#cellule en ordonnée   
dlanguesy={}

for file in list(d_act.keys()):
    language=d_act[file]['label']
    if language not in dlanguesy:
        dlanguesy[language]=[]
    activ=float(d_act[file]['activations'][ylstm][ycelltype][ycellnumber])
    dlanguesy[language].append(activ)

#Figure (fichiers hors de la base de F.Ramus)
for language in dlanguesy:
    std=np.std(dlanguesx[language])
    xsterr=std/(math.sqrt(len(dlanguesx[language])))
    std=np.std(dlanguesy[language])
    ysterr=std/(math.sqrt(len(dlanguesy[language])))
    x=np.mean(dlanguesx[language])
    y=np.mean(dlanguesy[language])
    if language in langues: #en bleu les langues qui existent dans la base de F. Ramus
        plt.errorbar(x,y,label=language,fmt=".b",xerr=xsterr,yerr=ysterr,capsize=2)
        plt.text(x+0.0001,y+0.0002,language)
    else: #en noir les autres langues
        plt.errorbar(x,y,label=language,fmt=".k",xerr=xsterr,yerr=ysterr,capsize=2)
        plt.text(x+0.0001,y+0.0002,language)

title='Distribution des langues selon '+xlabel+', '+ylabel
plt.title(title)
plt.ylabel(ylabel)
plt.xlabel(xlabel)
name='distrib_'+xlabel.split(' ')[2]+'_'+ylabel.split(' ')[2]+'_ramus.pdf'
plt.savefig(name)
plt.close()

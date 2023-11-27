##Récupérer les 154 fichiers d'activation
import os
import sys
import json

saveTxtFile = True
saveFigures = True

os.chdir("../activations/Scores_Ramus/weights0904-2d-30Hz-newinputs-dataaugmentation-30epochs")

files=os.listdir("./")

d_act={}

#Liste et dictionnaires utilisés :
# - d_match={fichier retranscrit : fichier audio}
# - datay={fichier audio : liste des 512 activations} / {fichier audio : {'cell_states' : liste des 256 activations, 'outputs' : liste des 256 activations}
# - d_act={fichier audio : data}
# - data={'filename':nom du fichier,'label'=langue correcte, 'predicted':langue prédite,'activations':{'lstm_1':{'outputs':[activations],'cell_states':[activations]},'lstm_2':{'outputs':[activations],'cell_states':[activations]}},'scores':{langue:probabilité prédite}}


#Version indifférenciée
"""
datay={}

for file in files:
    with open(file,'r') as json_file:
        data=json.load(json_file)
    d_act[file]=data
    datay[data['filename']]=[]
    for lstm in data['activations']:
        for celltype in data['activations'][lstm]:
            for i in data['activations'][lstm][celltype]:
                datay[data['filename']].append(i)
"""

#Version différenciée mémoire/output et selon les couches

datay={}

for file in files:
    with open(file,'r') as json_file:
        data=json.load(json_file)
    d_act[file]=data
    datay[data['filename']]={'lstm_1':{'cell_states':[],'outputs':[]},'lstm_2':{'cell_states':[],'outputs':[]}}
    for lstm in data['activations']:
        for celltype in data['activations'][lstm]:
            for i in data['activations'][lstm][celltype]:
                datay[data['filename']][lstm][celltype].append(i)


os.chdir("../../..")
   
##Lecture & calcul base Ramus
os.chdir("./compar/corpus_ramus/Files")
files=os.listdir("./")
files.remove("ESP1167.LBN")
#files.remove(".DS_Store")
#files.remove('DUL1151.LBN')

import numpy as np
import math

phonemes=['e', 'L', 'm', 'a', 'i', 'v', 't', 'E', 'n', 'r', 'l', 'p', 'o', 's', 'b', 'd', 'k', 'f', 'u', 'Z', 'O', 'z', 'j', 'w', 'g', 'rr', 'D', 'x', 'B', '@', 'S', '2', 'Ei', 'R', 'h', 'y', 'G', 'A', 'I', '9y', 'Au', 'N', '9', 'aU', 'ei', 'ai', 'A:', 'T', 'O:', 'tS', '3:', 'ou', 'Oi', 'dZ', 'J', 'e~', 'o~', 'a~', 'H', 'LL', 'tt', 'ddz', 'ts', 'll', 'ddZ', 'nn', 'tts', 'kk', 'ss', 'mm', 'dz']

voyelles=['e','a','i','E','o','u','O','@','A','I','Au','aU','ei','ai','A:','O:','3:','ou','Oi','e~','o~','a~','y','2','9','Ei','9y']
consonnes=['L','m','v','t','n','l','p','s','b','d','k','f','Z','z','g','rr','D','x','B','S','R','h','G','N','T','tS','dZ','J','LL','tt','ll','r','ddz','ts','ddZ','nn','tts','kk','ss','mm','dz']
glides=['j','w','H']

d_deltaV,d_deltaC,d_propV={},{},{}
dfiles,d={},{}


# - d={fichier : [pour chaque ligne[phoneme,echantillonage,temps]]}
# + locale par fichier : Vduree=[durée de chaque intervalle de voyelles] (idem pr Cduree)
# - dfiles={fichier : [delta_V,delta_C,prop_V]}


#Sans différenciation des langues
"""
for file in files:
    L=[]
    f=open(file,"r")
    for line in f.readlines():
        l=line[:-1].split("\t")
        L.append(l)
    d[file]=L
    f.close()

def longueurs_intervalles(file):
    Vduree, Cduree=[],[]
    ligne=0
    while ligne+1<len(d[file]):
        v=0
        while ligne+1<len(d[file]) and (d[file][ligne][0] in voyelles or (ligne!=0 and d[file][ligne][0] in glides and d[file][ligne-1][0] in voyelles and d[file][ligne+1][0] not in voyelles)) and float(d[file][ligne][2])<3.0:: #conditions sur les semi voyelles + voyelles
        
            v+=min(3.0,float(d[file][ligne+1][2])-float(d[file][ligne][2]))
            ligne+=1
        if v!=0:
            Vduree.append(v)
        
        c=0
        while ligne+1<len(d[file]) and (d[file][ligne][0] in consonnes or (d[file][ligne][0] in glides and d[file][ligne+1][0] in voyelles)) and float(d[file][ligne][2])<3.0: #conditions sur les consonnes + semi voyelles
        
            c+=min(3.0,float(d[file][ligne+1][2])-float(d[file][ligne][2]))
            ligne+=1
        if c!=0:
            Cduree.append(c)
            
        elif c==0 and v==0:
            ligne+=1
    return Vduree, Cduree

def calcul_metriques(file):
    Vduree,Cduree=longueurs_intervalles(file)
    
    moyenneV,varianceV=np.mean(Vduree),0
    for j in Vduree:
        varianceV+=(j-moyenneV)**2
    varianceV=varianceV/(len(Vduree)-1)
    
    moyenneC,varianceC=np.mean(Cduree),0
    for j in Cduree:
        varianceC+=(j-moyenneC)**2
    varianceC=varianceC/(len(Cduree)-1)
    
    delta_V=math.sqrt(varianceV)
    delta_C=math.sqrt(varianceC)
    propV=(sum(Vduree)/(sum(Vduree)+sum(Cduree)))*100
    
    return delta_V, delta_C, propV
    
for file in d:
        delta_V, delta_C, propV=calcul_metriques(file)        
        dfiles[file]=[delta_V,delta_C,propV]
"""

#Avec différenciation des langues

# - d_langue={fichier : [pour chaque ligne[phoneme,echantillonage,temps]]}
# - D={langue : d_langue}
# - d_deltaV={langue : [delta_V de chaque fichier]} (idem pr d_propV et d_deltaC)

d_cat,d_du,d_en,d_esp,d_fr,d_it,d_ja,d_pol={},{},{},{},{},{},{},{}

for file in files:
    L=[]
    f=open(file,"r")
    for line in f.readlines():
        l=line[:-1].split("\t")
        L.append(l)
    d[file]=L
    if file[:3]=="CAT":
        d_cat[file]=L
    elif file[:3]=="DUL":
        d_du[file]=L
    elif file[:3]=="ENL":
        d_en[file]=L
    elif file[:3]=="ESP":
        d_esp[file]=L
    elif file[:3]=="FRL":
        d_fr[file]=L
    elif file[:3]=="ITL":
        d_it[file]=L
    elif file[:3]=="JAL":
        d_ja[file]=L
    elif file[:3]=="POL":
        d_pol[file]=L
    f.close()

D={"en":d_en,"pol":d_pol,"du":d_du,"fr":d_fr,"esp":d_esp,"it":d_it,"cat":d_cat,"ja":d_ja}


def longueurs_intervalles(file):
    Vduree, Cduree=[],[]
    ligne=0
    while ligne+1<len(d[file]):
        v=0
        while ligne+1<len(d[file]) and (d[file][ligne][0] in voyelles or (ligne!=0 and d[file][ligne][0] in glides and d[file][ligne-1][0] in voyelles and d[file][ligne+1][0] not in voyelles)) and float(d[file][ligne][2])<3.0: #conditions sur les semi voyelles + voyelles
        
            v+=min(3.0,float(d[file][ligne+1][2])-float(d[file][ligne][2]))
            ligne+=1
        if v!=0:
            Vduree.append(v)
        
        c=0
        while ligne+1<len(d[file]) and (d[file][ligne][0] in consonnes or (d[file][ligne][0] in glides and d[file][ligne+1][0] in voyelles)) and float(d[file][ligne][2])<3.0: #conditions sur les consonnes + semi voyelles
        
            c+=min(3.0,float(d[file][ligne+1][2])-float(d[file][ligne][2]))
            ligne+=1
        if c!=0:
            Cduree.append(c)
            
        elif c==0 and v==0:
            ligne+=1
    return Vduree, Cduree

def calcul_metriques(file):
    Vduree,Cduree=longueurs_intervalles(file)
    
    moyenneV,varianceV=np.mean(Vduree),0
    for j in Vduree:
        varianceV+=(j-moyenneV)**2
    varianceV=varianceV/(len(Vduree)-1)
    
    moyenneC,varianceC=np.mean(Cduree),0
    for j in Cduree:
        varianceC+=(j-moyenneC)**2
    varianceC=varianceC/(len(Cduree)-1)
    
    delta_V=math.sqrt(varianceV)
    delta_C=math.sqrt(varianceC)
    propV=(sum(Vduree)/(sum(Vduree)+sum(Cduree)))*100
    
    return delta_V, delta_C, propV
    

for langue in D:
    d_deltaV[langue]=[]
    d_deltaC[langue]=[]
    d_propV[langue]=[]

    for file in D[langue]:
        if file in d:
            delta_V, delta_C, propV=calcul_metriques(file)
            
            d_deltaV[langue].append(delta_V)
            d_deltaC[langue].append(delta_C)
            d_propV[langue].append(propV)
            
            dfiles[file]=[delta_V,delta_C,propV]

##Matcher les fichiers audio-retranscris :
audiofiles,writtenfiles=list(d_act.keys()),list(d.keys())
d_match={}
for file in writtenfiles:
    name=file.split('.')[0].upper()
    for audio in audiofiles:
        if name in audio.upper():
            d_match[file]=audio
            
##Calcul du pearson
import matplotlib.pyplot as plt
import scipy.stats.stats as scp

os.chdir("../..")

pearson={}

for i in ['deltC','deltV','propV']:
    pearson[i]={'lstm_1':{'cell_states':[],'outputs':[]},'lstm_2':{'cell_states':[],'outputs':[]}}

for lstm in ['lstm_1','lstm_2']:
    for celltype in ['outputs','cell_states']:
        for i in range(128): #i passe sur chacune des cellules
            xdeltV,xdeltC,xpropV,y=[],[],[],[] #x et y vont collecter les métriques et les données calculées par les neurones des 153 fichiers
            for file in list(d_match.keys()):
                xdeltV.append(dfiles[file][0])
                xdeltC.append(dfiles[file][1])
                xpropV.append(dfiles[file][2])
                audio=d_match[file].split('.')[0]
                y.append(datay[audio][lstm][celltype][i])
            y=np.array(y).astype(np.float)
    
            pearson['deltV'][lstm][celltype].append(math.sqrt(scp.pearsonr(y,xdeltV)[0]**2))
            pearson['deltC'][lstm][celltype].append(math.sqrt(scp.pearsonr(y,xdeltC)[0]**2))
            pearson['propV'][lstm][celltype].append(math.sqrt(scp.pearsonr(y,xpropV)[0]**2))

##Meilleures cellules
d1={}
for metrique in list(pearson.keys()):
    d1[metrique]=[]
    for lstm in list(pearson[metrique].keys()):
        for celltype in list(pearson[metrique][lstm].keys()):
            d1[metrique]=d1[metrique]+pearson[metrique][lstm][celltype]


if saveTxtFile:
	f=open('./corpus_ramus/best_cells.txt', 'w')
	orig_stdout = sys.stdout
	sys.stdout=f

for metrique in list(pearson.keys()):
    print(metrique)
    for i in range(5):
        if i!=0:
            d1[metrique].remove(m)
        m=max(d1[metrique])
        for lstm in list(pearson[metrique].keys()):
            for celltype in list(pearson[metrique][lstm].keys()):
                if m in pearson[metrique][lstm][celltype]:
                    index=pearson[metrique][lstm][celltype].index(m)
                    text=lstm+", "+celltype+", "+str(index)+'\t\t'
                    print(text,m)
    print("\n")


if saveTxtFile:
	f.close()
	sys.stdout=orig_stdout



##Figures
#delta V

if saveFigures:
	os.chdir('./corpus_ramus/Figures')

	plt.hist(pearson['deltV']['lstm_1']['outputs'])
	plt.title('Coef de corrélation des cellules output de la lstm1, métrique delta V')
	plt.savefig('pearson_lstm1_output_deltav.pdf')
	plt.close()

	plt.hist(pearson['deltV']['lstm_1']['cell_states'])
	plt.title('Coef de corrélation des cellules mémoire de la lstm1, métrique delta V')
	plt.savefig('pearson_lstm1_cellstates_deltav.pdf')
	plt.close()

	plt.hist(pearson['deltV']['lstm_2']['outputs'])
	plt.title('Coef de corrélation des cellules output de la lstm2, métrique delta V')
	plt.savefig('pearson_lstm2_output_deltav.pdf')
	plt.close()

	plt.hist(pearson['deltV']['lstm_2']['cell_states'])
	plt.title('Coef de corrélation des cellules mémoire de la lstm2, métrique delta V')
	plt.savefig('pearson_lstm2_cellstates_deltav.pdf')
	plt.close()

	#delta C
	plt.hist(pearson['deltC']['lstm_1']['outputs'])
	plt.title('Coef de corrélation des cellules output de la lstm1, métrique delta C')
	plt.savefig('pearson_lstm1_output_deltac.pdf')
	plt.close()

	plt.hist(pearson['deltC']['lstm_1']['cell_states'])
	plt.title('Coef de corrélation des cellules mémoire de la lstm1, métrique delta C')
	plt.savefig('pearson_lstm1_cellstates_deltac.pdf')
	plt.close()

	plt.hist(pearson['deltC']['lstm_2']['outputs'])
	plt.title('Coef de corrélation des cellules output de la lstm2, métrique delta C')
	plt.savefig('pearson_lstm2_output_deltac.pdf')
	plt.close()

	plt.hist(pearson['deltC']['lstm_2']['cell_states'])
	plt.title('Coef de corrélation des cellules mémoire de la lstm2, métrique delta C')
	plt.savefig('pearson_lstm2_cellstates_deltac.pdf')
	plt.close()

	#prop V
	plt.hist(pearson['propV']['lstm_1']['outputs'])
	plt.title('Coef de corrélation des cellules output de la lstm1, métrique prop V')
	plt.savefig('pearson_lstm1_output_propv.pdf')
	plt.close()

	plt.hist(pearson['propV']['lstm_1']['cell_states'])
	plt.title('Coef de corrélation des cellules mémoire de la lstm1, métrique prop V')
	plt.savefig('pearson_lstm1_cellstates_propv.pdf')
	plt.close()

	plt.hist(pearson['propV']['lstm_2']['outputs'])
	plt.title('Coef de corrélation des cellules output de la lstm2, métrique prop V')
	plt.savefig('pearson_lstm2_output_propv.pdf')
	plt.close()

	plt.hist(pearson['propV']['lstm_2']['cell_states'])
	plt.title('Coef de corrélation des cellules mémoire de la lstm2, métrique prop V')
	plt.savefig('pearson_lstm2_cellstates_propv.pdf')
	plt.close()

	os.chdir('../..')
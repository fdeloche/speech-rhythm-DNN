#D apres calcul_pearson.py

##Récupérer les 154 fichiers d'activation
import os
import sys
import json

import numpy as np
#import math


def create_dictionaries(activations_folder='../activations/Scores_Ramus/weights0904-2d-30Hz-newinputs-dataaugmentation-30epochs',
	metrics_folder='./corpus_ramus/Files/'):
	'''
	Returns:
		dict: dfiles {fichier : [delta_V,delta_C,prop_V]} (Ramus' metrics)
		dict: d_match {fichier retranscrit : fichier audio (noms)}. allows to go from dfiles to datay
		dict: datay {nom fichier audio : liste des 512 activations} / {fichier audio : {'cell_states' : liste des 256 activations, 'outputs' : liste des 256 activations}
		dict: d_act {nom fichier audio : data}  (same but with other data)
		dict: d_lang {fichier retranscrit : langue (label)}
		dict: D {langue : d_langue}  data corpus Ramus. d_langue={file : [for every line [phoneme,echantillonage,temps]]}
		dict: dmetrics {fichier: dict metrics}  (additional metrics for Ramus data)
	'''

	files=os.listdir(activations_folder)

	d_act={}


	#Version différenciée mémoire/output et selon les couches

	datay={}

	for file0 in files:
		file=f'{activations_folder}/{file0}'
		with open(file,'r') as json_file:
			data=json.load(json_file)
		d_act[file0]=data
		datay[data['filename']]={'lstm_1':{'cell_states':[],'outputs':[]},'lstm_2':{'cell_states':[],'outputs':[]}}
		for lstm in data['activations']:
			for celltype in data['activations'][lstm]:
				for i in data['activations'][lstm][celltype]:
					datay[data['filename']][lstm][celltype].append(i)



	files=os.listdir(metrics_folder)
	files.remove("ESP1167.LBN")
	#files.remove(".DS_Store")
	#files.remove('DUL1151.LBN')


	phonemes=['e', 'L', 'm', 'a', 'i', 'v', 't', 'E', 'n', 'r', 'l', 'p', 'o', 's', 'b', 'd', 'k', 'f', 'u', 'Z', 'O', 'z', 'j', 'w', 'g', 'rr', 'D', 'x', 'B', '@', 'S', '2', 'Ei', 'R', 'h', 'y', 'G', 'A', 'I', '9y', 'Au', 'N', '9', 'aU', 'ei', 'ai', 'A:', 'T', 'O:', 'tS', '3:', 'ou', 'Oi', 'dZ', 'J', 'e~', 'o~', 'a~', 'H', 'LL', 'tt', 'ddz', 'ts', 'll', 'ddZ', 'nn', 'tts', 'kk', 'ss', 'mm', 'dz']

	voyelles=['e','a','i','E','o','u','O','@','A','I','Au','aU','ei','ai','A:','O:','3:','ou','Oi','e~','o~','a~','y','2','9','Ei','9y']
	consonnes=['L','m','v','t','n','l','p','s','b','d','k','f','Z','z','g','rr','D','x','B','S','R','h','G','N','T','tS','dZ','J','LL','tt','ll','r','ddz','ts','ddZ','nn','tts','kk','ss','mm','dz']
	glides=['j','w','H']

	d_deltaV,d_deltaC,d_propV={},{},{}
	dfiles,d, dmetrics={},{}, {}


	# - d={fichier : [pour chaque ligne[phoneme,echantillonage,temps]]}
	# + locale par fichier : Vduree=[durée de chaque intervalle de voyelles] (idem pr Cduree)
	# - dfiles={fichier : [delta_V,delta_C,prop_V]}


	#Avec différenciation des langues

	# - d_langue={fichier : [pour chaque ligne[phoneme,echantillonage,temps]]}
	# - D={langue : d_langue}
	# - d_deltaV={langue : [delta_V de chaque fichier]} (idem pr d_propV et d_deltaC)

	d_cat,d_du,d_en,d_esp,d_fr,d_it,d_ja,d_pol={},{},{},{},{},{},{},{}

	for file0 in files:

		file=f'{metrics_folder}/{file0}'
		L=[]
		f=open(file,"r")
		for line in f.readlines():
			l=line[:-1].split("\t")
			L.append(l)
		d[file0]=L
		if file0[:3]=="CAT":
			d_cat[file0]=L
		elif file0[:3]=="DUL":
			d_du[file0]=L
		elif file0[:3]=="ENL":
			d_en[file0]=L
		elif file0[:3]=="ESP":
			d_esp[file0]=L
		elif file0[:3]=="FRL":
			d_fr[file0]=L
		elif file0[:3]=="ITL":
			d_it[file0]=L
		elif file0[:3]=="JAL":
			d_ja[file0]=L
		elif file0[:3]=="POL":
			d_pol[file0]=L
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
		
		delta_V=np.sqrt(varianceV)
		delta_C=np.sqrt(varianceC)
		propV=(sum(Vduree)/(sum(Vduree)+sum(Cduree)))*100
		
		#additional metrics

		#VARCOS
		varco_V=np.sqrt(varianceV)/moyenneV*100
		varco_C=np.sqrt(varianceC)/moyenneC*100

		#PVIs
		nPVI_V=0
		for k in range(len(Vduree)-1):
			num=np.abs(Vduree[k+1]-Vduree[k])
			den=(Vduree[k+1]+Vduree[k])/2
			nPVI_V+=num/den
		nPVI_V/=(len(Vduree)-1)
		nPVI_V*=100

		rPVI_C=0
		for k in range(len(Cduree)-1):
			num=np.abs(Cduree[k+1]-Cduree[k])
			#den=(Cduree[k+1]+Cduree[k])/2   #no normalization for consonants
			rPVI_C+=num
		rPVI_C/=(len(Vduree)-1)
		rPVI_C*=100

		return delta_V, delta_C, propV,  {'deltV':delta_V, 'deltC': delta_C, 'propV': propV, 
				'varco_V':varco_V, 'varco_C':varco_C, 'nPVI_V':nPVI_V, 'rPVI_C':rPVI_C}


		
	dlang={}
	for langue in D:
		d_deltaV[langue]=[]
		d_deltaC[langue]=[]
		d_propV[langue]=[]

		for file in D[langue]:
			if file in d:
				delta_V, delta_C, propV, add_metrics=calcul_metriques(file)
				
				d_deltaV[langue].append(delta_V)
				d_deltaC[langue].append(delta_C)
				d_propV[langue].append(propV)
				
				dfiles[file]=[delta_V,delta_C,propV]
				dlang[file]=langue
				dmetrics[file]=add_metrics


	##Matcher les fichiers audio-retranscris :
	audiofiles,writtenfiles=list(d_act.keys()),list(d.keys())
	d_match={}
	for file in writtenfiles:
		name=file.split('.')[0].upper()
		for audio in audiofiles:
			if name in audio.upper():
				d_match[file]=audio

	return dfiles, d_match, datay, d_act, dlang, D, dmetrics
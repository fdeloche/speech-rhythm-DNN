##Lecture de la base
import os
import sys

createFileMetrics=False
createFigures=False

os.chdir("./Files")
files=os.listdir(".")
files.remove("ESP1167.LBN")
#files.remove(".DS_Store")

d={}
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

##Phonèmes
import numpy as np
import matplotlib.pyplot as plt
import math

phonemes=['e', 'L', 'm', 'a', 'i', 'v', 't', 'E', 'n', 'r', 'l', 'p', 'o', 's', 'b', 'd', 'k', 'f', 'u', 'Z', 'O', 'z', 'j', 'w', 'g', 'rr', 'D', 'x', 'B', '@', 'S', '2', 'Ei', 'R', 'h', 'y', 'G', 'A', 'I', '9y', 'Au', 'N', '9', 'aU', 'ei', 'ai', 'A:', 'T', 'O:', 'tS', '3:', 'ou', 'Oi', 'dZ', 'J', 'e~', 'o~', 'a~', 'H', 'LL', 'tt', 'ddz', 'ts', 'll', 'ddZ', 'nn', 'tts', 'kk', 'ss', 'mm', 'dz']

voyelles=['e','a','i','E','o','u','O','@','A','I','Au','aU','ei','ai','A:','O:','3:','ou','Oi','e~','o~','a~','y','2','9','Ei','9y']
consonnes=['L','m','v','t','n','l','p','s','b','d','k','f','Z','z','g','rr','D','x','B','S','R','h','G','N','T','tS','dZ','J','LL','tt','ll','r','ddz','ts','ddZ','nn','tts','kk','ss','mm','dz']
glides=['j','w','H']

d_deltaV,d_deltaC,d_propV={},{},{}
D={"en":d_en,"pol":d_pol,"du":d_du,"fr":d_fr,"esp":d_esp,"it":d_it,"cat":d_cat,"ja":d_ja}
d_Vi,d_Ci={},{}


##Calcul des métriques

#Listes et dictionnaires utilisés :
# - d={fichier : [pour chaque ligne[phoneme,echantillonage,temps]]}
# - d_langue={fichier : [pour chaque ligne[phoneme,echantillonage,temps]]}
# - D={langue : d_langue}
# - d_deltaV={langue : [delta_V de chaque fichier]} (idem pr d_propV et d_deltaC)
# - d_Vi={langue : nombre d'intervales de voyelles} (idem pr d_Ci)
# + locale par fichier : Vduree=[durée de chaque intervalle de voyelles] (idem pr Cduree)

def longueurs_intervalles(file):
	Vduree, Cduree=[],[]
	ligne=0
	while ligne+1<len(d[file]):
		v=0
		while ligne+1<len(d[file]) and (d[file][ligne][0] in voyelles or (ligne!=0 and d[file][ligne][0] in glides and d[file][ligne-1][0] in voyelles and d[file][ligne+1][0] not in voyelles)): #conditions sur les semi voyelles + voyelles
			v+=float(d[file][ligne+1][2])-float(d[file][ligne][2])
			ligne+=1
		if v!=0:
			Vduree.append(v)
		
		c=0
		while ligne+1<len(d[file]) and (d[file][ligne][0] in consonnes or (d[file][ligne][0] in glides and d[file][ligne+1][0] in voyelles)): #conditions sur les consonnes + semi voyelles
			c+=float(d[file][ligne+1][2])-float(d[file][ligne][2])
			ligne+=1
		if c!=0:
			Cduree.append(c)
			
		elif c==0 and v==0:
			ligne+=1
	Vi,Ci=len(Vduree),len(Cduree) #nombre d'intervalles de voyelles/consonnes
	return Vduree, Cduree, Vi, Ci

def calcul_metriques(file):
	Vduree=longueurs_intervalles(file)[0]
	Cduree=longueurs_intervalles(file)[1]
	
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
	
	Vi,Ci=0,0

	for file in D[langue]:
		delta_V, delta_C, propV=calcul_metriques(file)
		
		d_deltaV[langue].append(delta_V)
		d_deltaC[langue].append(delta_C)
		d_propV[langue].append(propV)
		
		Vi+=longueurs_intervalles(file)[2]
		Ci+=longueurs_intervalles(file)[3]
		
	d_Vi[langue]=Vi
	d_Ci[langue]=Ci
		
##Tableau de valeurs
if createFileMetrics:
	f=open('../Ramus_metrics.txt', 'w')
	orig_stdout = sys.stdout
	sys.stdout=f


print("langue\tV intervals\tC intervals\t%V\tdelta V\tdelta C\n")
for langue in D:
	print(langue[:2],"\t\t",d_Vi[langue],"\t\t",d_Ci[langue],"\t",round(np.mean(d_propV[langue]),1),"\t",round(np.mean(d_deltaV[langue])*100,2),"\t",round(np.mean(d_deltaC[langue])*100,2))

if createFileMetrics:
	f.close()
	sys.stdout=orig_stdout

if createFigures:
	##Figures
	os.chdir("../Figures")

	for i in D:
		std=np.std(d_propV[i])
		xsterr=std/math.sqrt(len(d_propV[i]))
		std=np.std(d_deltaC[i])
		ysterr=std/(math.sqrt(len(d_deltaC[i])))
		x,y=np.mean(d_propV[i]),np.mean(d_deltaC[i])
		plt.errorbar(x,y,label=i,fmt=".k",xerr=xsterr,yerr=ysterr,capsize=2)
		plt.text(x+0.25,y+0.0005,i)

	plt.title("Distribution des langues selon %V, delta C")
	plt.ylabel("delta C")
	plt.xlabel("%V")
	plt.savefig("ramus_fig1.1.pdf")
	plt.close()

	for i in D:
		std=np.std(d_propV[i])
		xsterr=std/math.sqrt(len(d_propV[i]))
		std=np.std(d_deltaV[i])
		ysterr=std/math.sqrt(len(d_deltaV[i]))
		x,y=np.mean(d_propV[i]),np.mean(d_deltaV[i])
		plt.errorbar(x,y,label=i,fmt=".k",xerr=xsterr,yerr=ysterr,capsize=2)
		plt.text(x+0.25,y+0.0005,i)
		
	plt.title("Distribution des langues selon %V, delta V")
	plt.ylabel("delta V")
	plt.xlabel("%V")
	plt.savefig("ramus_fig1.2.pdf")
	plt.close()

	for i in D:
		std=np.std(d_deltaV[i])
		xsterr=std/math.sqrt(len(d_deltaV[i]))
		std=np.std(d_deltaC[i])
		ysterr=std/math.sqrt(len(d_deltaC[i]))
		x,y=np.mean(d_deltaV[i]),np.mean(d_deltaC[i])
		plt.errorbar(x,y,label=i,fmt=".k",xerr=xsterr,yerr=ysterr,capsize=2)
		plt.text(x+0.0005,y+0.0005,i)

	plt.title("Distribution des langues selon delta V, delta C")
	plt.ylabel("delta C")
	plt.xlabel("delta V")
	plt.savefig("ramus_fig1.3.pdf")
	plt.close()
	os.chdir('..')

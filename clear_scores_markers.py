import sys
import os

def clear_markers(languages, nFolder="./Normalized_Data"):
	'''clear markers in folder nFolder (normalized data)'''
	print("Clear markers for languages : ")
	print(languages)

	for l in languages:
	    for root, subFolders, files in os.walk("{}/{}".format(nFolder, l)):
	        for f in files:
	            #if .MARKER
	            if (f.split('.')[-1]=="marker"):
	            	os.remove("{}/{}".format(root, f))

if __name__=='__main__':
	languages = sys.argv[1::]

	if "all" in languages:
		print("autodect available languages")
		languages = []
		for dir_ in os.listdir("./Files"):
			if(os.path.isdir("./Files/{}".format(dir_))):
				languages.append(dir_)

	clear_markers(languages)
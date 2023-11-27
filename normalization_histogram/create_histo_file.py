import timit as tm
import numpy as np

import matplotlib.pyplot as pl

save_histo_file=True

n_files=512
w_size=512

timitFiles = tm.listTimitFiles('/home/fdeloche/Dropbox/SpeechCode2/corpus/TIMIT/TIMIT', pick=n_files)
rms_list = []
for tmFile_str in timitFiles:
    tmFile = tm.TimitFile(tmFile_str)
    tmFile.extract_signal()
    s = tmFile.get_signal()
    s = s[0:np.size(s)//w_size*w_size] #XXX int division
    s_rs = np.reshape(s, (-1, w_size))
    rms = np.sqrt(np.mean(s_rs**2, axis=1)) #RMS
    rms_list.append(rms)


rms_concat = np.concatenate(rms_list)

#ignore 60% lower
rms_concat=np.sort(rms_concat)
n_values=len(rms_concat)
rms_concat=rms_concat[int(n_values*0.6):]

c_cor = np.log(10)
m = 20

bins = np.linspace(0.02, 0.15, m+1)


(hist, bins) = np.histogram(rms_concat, bins=bins)
hist = hist*1./np.sum(hist)

histo_file = "normalization_histogram.npz"


pl.figure()
pl.title('Histogram amplitude')
pl.hist(rms_concat, bins=bins)
pl.show()


if save_histo_file:
    np.savez(histo_file, hist=hist, bins=bins)
    print("Histogram file created at {}".format(histo_file))


bins = np.linspace(-220., 0, m+1)/c_cor
rms_concat_db = 20*np.log10(rms_concat)
(hist, bins) = np.histogram(rms_concat, bins=bins)
histo_file_db = "normalization_histogram_db.npz"

if save_histo_file:
    np.savez(histo_file, hist=hist, bins=bins)
    print("Histogram file created at {}".format(histo_file_db))
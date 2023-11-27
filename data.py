'''
Functions for creating/retrieving data.
See also input.py for the creation of the dataset with TFrecords.
'''
import tensorflow as tf
import numpy as np
from scipy.signal import tukey, lfilter
import parselmouth

class ScoresComputer:
	def __init__(self, fs=16000, w_size=256, step=256, verbose=True, 
		computeRMSValue=True, windowing=True, margin=16, 
		computeF0=True, pitch_floor=75, pitch_ceiling=500,
		computeHPRMSValue=True, fcutb=200, fcuta=5000):
		'''
		:param fs: sampling frequency of audio inputs
		:param w_size: window size (computations of RMS values) in number of samples
		:param step: step size (in number of samples). overlap between succesive windows is (1-step/window_size). sampling frequency of scores is fs/size (ex: 16000Hz -> 256, 60Hz, 512, 30Hz)
		:param computeRMSValue: if True, will compute RMS values. Key: 'rmsValue'
		:param windowing: If True, use a Tukey window for computation of RMS with margins defined by margin (margin < w_size)
		:param computeF0: if True, compute fundamental frequency with Praat algorithm (see Praat doc for relevance of pitch_floor and pitch_ceiling). Key: 'F0'
		:param computeHPRMSValue: If True, compute RMS values for a high-pass version of the signal (HP fcutb, LP fcuta), see Fant et al. 2000 study of prominence. Key 'HRmsValue'
		'''
		self.fs=fs
		self.w_size = w_size
		self.step=step
		self.out_fs=fs/w_size


		self.computeRMSValue=computeRMSValue
		self.windowing=windowing
		self.margin=margin
		self.alpha_Tuckey=2*margin/w_size
		self.tuckeyWindow=tukey(w_size+2*margin, alpha=self.alpha_Tuckey)
		add_text=""
		alpha_Tuckey_percent=self.alpha_Tuckey*100
		if windowing:
			add_text=f". Overlap with Tuckey windows (alpha={alpha_Tuckey_percent:.1f} %)"

		self.computeF0 = computeF0
		self.pitch_floor = pitch_floor
		self.pitch_ceiling = pitch_ceiling

		self.computeHPRMSValue=computeHPRMSValue
		self.fcutb = fcutb
		self.fcuta = fcuta

		coeff=2*fs
		B=coeff*1/(2*np.pi*fcutb)
		A=coeff*1/(2*np.pi*fcuta)

		self.filt_b=np.array([B+1, 1-B])
		self.filt_a=np.array([A+1, 1-A])

		print(f"Init scoresComputer, output fs: {self.out_fs:.1f} Hz")
		print("features: ")
		if self.computeRMSValue:
			print(f" - RMS value, win size: {w_size}, step size: {step}, {add_text}")
		if self.computeF0:
			print(f" - F0 (Praat) : pitch floor {pitch_floor:.0f} Hz, pitch ceiling {pitch_ceiling:.0f} Hz")
		if self.computeHPRMSValue:
			print(f" - HP + RMS value, f_cut_b = {fcutb:.0f} Hz (HP), f_cut_a = {fcuta:.0f} Hz (LP)")

	def compute_scores(self, s):
		'''computes the scores for input s. Output is a dict of 1D array {rmsValue}'''

		m = np.size(s)

		#for RMS values
		step = self.step
		m-=m%self.w_size #must be a multiple of w_size
		n_bins = m//step

		scores = {}

		#RMS values
		computeRMST = self.computeRMSValue, self.computeHPRMSValue
		namesT='rmsValue', 'HRmsValue'
		filteredT=False, True

		for computeRMS, name, filtered in zip(computeRMST, namesT, filteredT):
			if computeRMS:
				scores[name]=np.zeros(n_bins)

				if filtered:
					s2=lfilter(self.filt_b, self.filt_a, s)
				else:
					s2=s

				for i in range(n_bins):
					useTuckey=self.windowing and not(i==0 or i==(n_bins-1)) #first bin or last bin don't use the Tuckey window
					samp_begin = step*i
					samp_end=samp_begin+self.w_size

					if useTuckey:
						win=s2[samp_begin-self.margin:samp_end+self.margin]
						win*=self.tuckeyWindow
					else:
						win = s2[samp_begin:samp_end]

					win=win-np.mean(win)
					win_rms=np.sqrt(np.mean(win**2))+0.1/np.frombuffer(b'\x7f\xff', dtype='>i2')
					scores[name][i]= win_rms

		if self.computeF0:
			snd=parselmouth.Sound(s, sampling_frequency=self.fs)
			pitch = snd.to_pitch(time_step=1/self.out_fs, pitch_floor=self.pitch_floor, 
				pitch_ceiling=self.pitch_ceiling)
			f0_arr=pitch.selected_array['frequency']
			#HACK zero pad f0_arr if smaller that n_bins
			diff_length=n_bins-len(f0_arr)
			assert (diff_length>=0 and diff_length<=2), f"difference of more than 2 between lengths of F0 arr and RMS arr ({diff_length})"
			if diff_length>=1:
				f0_arr=np.append(f0_arr, 0.)
			if diff_length==2:
				f0_arr=np.insert(f0_arr, 0, 0.)
			scores["F0"]=f0_arr
		return scores


def serialize_array(arr):
	return arr.astype(np.float32).tostring()

def createExample(**kwargs):
	'''creates a tf.train.Example from features dict'''
	features_dict={}
	for key, value in kwargs.items():
		if key in ["filename", "language", "database", "speaker"]: #str
			value=[value.encode('utf-8')]
		if key in ['rmsValue', 'HRmsValue', 'F0']: #float arr
			value=[serialize_array(value)]
		features_dict[key]=tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
	return tf.train.Example(features=tf.train.Features(feature=features_dict))


def createFeaturesDescription(rmsValue=True, filename=True, language=True, database=True, speaker=True, 
		HRmsValue=True, F0=True):
	'''Creates a feature description for reading the TFRecords'''
	features_description={}
	if rmsValue:
		features_description["rmsValue"]=tf.io.FixedLenFeature([], dtype=tf.string)
	if filename:
		features_description["filename"]=tf.io.FixedLenFeature([], dtype=tf.string)
	if language:
		features_description["language"]=tf.io.FixedLenFeature([], dtype=tf.string)
	if database:
		features_description["database"]=tf.io.FixedLenFeature([], dtype=tf.string)
	if speaker:
		features_description["speaker"]=tf.io.FixedLenFeature([], dtype=tf.string)
	if HRmsValue:
		features_description["HRmsValue"]=tf.io.FixedLenFeature([], dtype=tf.string)
	if F0:
		features_description["F0"]=tf.io.FixedLenFeature([], dtype=tf.string)
	return features_description

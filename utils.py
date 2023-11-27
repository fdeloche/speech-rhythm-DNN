import numpy as np
import os
from threading import Thread, Event
import functools


#UTILS
def listFiles(str_dir, pick=-1):
	l = []
	for root, subFolders, files in os.walk(str_dir, followlinks=True):
		for f in files:
			l.append(os.path.join(root, f))
	#print("number of files : " + str(len(l)))
	if pick==-1 or pick==np.inf:
		return l
	else:
		pick = min(len(l), pick)
		return random.sample(l, pick)

class LoadingBar(Thread):
	def __init__(self):
		super(LoadingBar, self).__init__()
		self.stopevent = Event()
		self.pauseevent = Event()
		self.pauseevent.set()
		self.counter=0
		self.counter_max=1

	def run(self):
		#Loading bar
		while not self.stopevent.wait(60):
			if not self.pauseevent.isSet():
				print("\r {} ({:.2f} %) files processed".format(self.counter, self.counter*100./self.counter_max), flush=True)

	def pause(self):
		self.pauseevent.set()

	def resume(self):
		self.pauseevent.clear()


	def stop(self):
		self.stopevent.set()


def lazy_property(function):
	attribute = '_cache_' + function.__name__

	@property
	@functools.wraps(function)
	def decorator(self):
		if not hasattr(self, attribute):
			setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return decorator

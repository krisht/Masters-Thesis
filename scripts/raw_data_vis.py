import mne
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
font = {'family' : 'FreeSerif',
        'size'   : 18}
plt.rc('text', usetex=True)
matplotlib.rc('font', **font)

from scipy import signal




eeg_dat = mne.io.read_raw_edf('./a_.edf', preload=True)


from collections import defaultdict
ch_types = defaultdict(str)
channel_names = []
channel_types = []

for ch in eeg_dat.ch_names:
    if 'EEG' in ch:
        ch_types[ch] = 'eeg'
    elif 'EMG' in ch:
        ch_types[ch] = 'emg'
    elif 'STI' in ch: 
        ch_types[ch] = 'stim'
    else:
        ch_types[ch] = 'misc'
        
eeg_dat.set_channel_types(ch_types)


# Example of one way to start fixing up the channels
#  Priority again is to keep everything consistent with MNE, so first check for MNE functions that can do 
#   things like this, and that can update this information insode the data.info object. 

from collections import defaultdict
ch_types = defaultdict(str)

channel_names = []
channel_types =[]

# lists to store the array names

for ch in eeg_dat.ch_names:    
    if 'EEG' in ch:
        ch_types[ch] = 'eeg'
        
    elif 'EMG' in ch:
        ch_types[ch] = 'emg'
        # Keeps track that this is an EMG channels
        
    elif 'STI' in ch:
        ch_types[ch] = 'stim'
        # Keeps track that this is an STI channel
    
    else:
        ch_types[ch] = 'misc'
        # Keeps track of MISC channels
        
eeg_dat.set_channel_types(ch_types)
        
for ch in eeg_dat.ch_names:    
    if 'EEG' in ch:
        ch = ch[4:7]
        ch =''.join(e for e in ch if e.isalnum())
        channel_names.append(ch)
        channel_types.append('eeg')
        ch_types[ch] = 'eeg'
        
    elif 'EMG' in ch:
        ch = ch[4:7]
        ch =''.join(e for e in ch if e.isalnum())
        channel_names.append(ch)
        channel_types.append('emg')
        ch_types[ch] = 'emg'
        # Keeps track that this is an EMG channels
        
    elif 'STI' in ch:
        channel_names.append(ch)
        channel_types.append('stim')
        ch_types[ch] = 'stim'
        # Keeps track that this is an STI channel
    
    else:
        channel_names.append(ch)
        channel_types.append('misc')
        ch_types[ch] = 'misc'
        # Keeps track of MISC channels
        
# The EEG channels use the standard naming strategy.
# By supplying the 'montage' parameter, approximate locations
montage = 'standard_1020'



raw = eeg_dat.plot()
raw.savefig('raw.pdf', bbox_inches='tight')
plt.close()

psd = eeg_dat.plot_psd()
psd.savefig('psd.pdf', bbox_inches='tight')
psd.title ='EEG Signal Power Spectral Density'
plt.close()

info = mne.create_info(channel_names, eeg_dat.info['sfreq'], channel_types, montage)
notches = np.arange(60, 61, 60)
eeg_dat.notch_filter(notches)
filtered = eeg_dat.copy().filter(1, 70, h_trans_bandwidth=10)
#print(filtered.get_data(start=0).shape)


for ii in range(30):
	plt.figure()
	X = filtered.get_data(start=42750, stop=43000)[ii,:]

	f, t, Sxx = signal.stft(X, 250, nperseg=1000)
	print(f.shape)
	print(t.shape)
	print(Sxx.shape)
	plt.pcolormesh(t, f, np.abs(Sxx), linewidth=0,rasterized=True)
	plt.title('STFT Magnitude')
	plt.ylim(ymin=0, ymax=71)
	plt.ylabel('Frequency $(Hz)$')
	plt.xlabel('Time $(s)$')
	plt.savefig('plot%d.pdf' % ii, bbox_inches='tight')
	plt.close()
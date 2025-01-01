import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from os import path, listdir
from scipy.signal import find_peaks

infile = path.join('wav', 'train_cdefg.wav')
outfile = None

a, fs  = sf.read(infile)
a = librosa.to_mono(a.T)
rHz = 10 #Frequency resolution in Hz
rs = 0.01 #Temporal resolution in seconds

# Perform FFT
D = librosa.stft(a, n_fft=fs//rHz, hop_length=int(fs*rs) )
freqs = librosa.fft_frequencies(sr=fs, n_fft=fs//rHz)
D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=8)
rp = np.max(np.abs(D))

fig, (ax1,ax2,ax3) = plt.subplots(3)
ax1.set_title("Harmonic spectrogram")
img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic), ref=rp), ax=ax1, y_axis='log')
fig.colorbar(img, ax=ax1)

ax2.set_title("Amplitude")
A_percussive = librosa.amplitude_to_db(np.sum(np.abs(D_percussive),axis=0))
ax2.plot(A_percussive, label = "percussive")

peaks,properties = find_peaks(A_percussive, height=45 )
print(f'Attacks found at {peaks}')

salience = librosa.salience(np.abs(D_harmonic), freqs=freqs, 
                            harmonics=[1,2,3,4,5,6,7,8])
salience = np.nan_to_num(salience)
ax2.plot(librosa.amplitude_to_db(np.max(salience, axis=0)), label="volume")

print(salience.shape)
ax3.set_title("Pitch")
ax3.plot([freqs[x] for x in np.argmax(salience, axis=0)])

if outfile is None:
    plt.show()
else:
    fig.savefig(outfile)
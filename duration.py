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

fig, (ax3,ax2) = plt.subplots(2)

A_percussive = librosa.amplitude_to_db(np.sum(np.abs(D_percussive),axis=0))

peaks,properties = find_peaks(A_percussive, height=45 )
print(f'Attacks found at {peaks}')
salience = librosa.salience(np.abs(D_harmonic), freqs=freqs, 
                            harmonics=list(range(1,8)))
salience = np.nan_to_num(salience)

ax3.set_title("Pitch")
pitch = [freqs[x] for x in np.argmax(salience, axis=0)]
peaks2 = np.append(peaks[1:], -1)
median_pitch = [np.median(pitch[x:y]) for x,y in zip(peaks, peaks2)]

padded_pitch = np.empty(D.shape[-1])
for i in range(len(median_pitch)):
    padded_pitch[peaks[i]:peaks2[i]] = median_pitch[i]
padded_pitch[:peaks[0]] = median_pitch[0]
timbre = librosa.f0_harmonics(D, freqs=freqs, f0=padded_pitch, harmonics=list(range(1,17)))
    
# Extend so it graphs correctly
median_pitch.append(median_pitch[-1])
ax3.step(np.append(peaks, len(A_percussive)),median_pitch, where="post")
ax3.set_xlim(0, len(A_percussive))

ax2.set_title("Timbre")
img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(timbre)), ax=ax2)
#fig.colorbar(img, ax=ax2)

if outfile is None:
    plt.show()
else:
    fig.savefig(outfile)
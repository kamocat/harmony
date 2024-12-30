import librosa
import numpy as np
import matplotlib.pyplot as plt
from os import path, listdir

def spectrum(infile, outfile=None):
    a, fs  = librosa.load(infile, sr=None, mono=True)
    print(a.shape)
    N = len(a)
    rHz = 10 #Frequency resolution in Hz
    rs = 0.01 #Temporal resolution in seconds

    # Perform FFT
    D = librosa.stft(a, n_fft=fs//rHz, hop_length=int(fs*rs) )
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, (ax1,ax2) = plt.subplots(2)
    ax1.set_title("Spectral intensity plot (librosa)")
    ax1.set(xlabel=f"Time (seconds * {rs})")
    img = librosa.display.specshow(S_db, ax=ax1)
    fig.colorbar(img, ax=ax1)

    ax2.set_title("Estimated pitch")
    pitches, magnitudes = librosa.piptrack(S=D)
    ax2.plot(pitches)
	
    if outfile is None:
        plt.show()
    else:
        fig.savefig(outfile)

d = 'wav'
e = 'img'
a = listdir('wav')
for x in a:
    infile = path.join(d,x)
    outfile = path.join(e,x.partition('.')[0]+'_librosa.png')
    print(x)
    spectrum(infile,outfile)

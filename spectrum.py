import librosa
import soundfile as sf
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
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=8)
    rp = np.max(np.abs(D))

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.set_title("Percussive spectrogram")
    img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_percussive), ref=rp), ax=ax1, y_axis='log')
    fig.colorbar(img, ax=ax1)

    ax2.set_title("Harmonic spectrogram")
    img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic), ref=rp), ax=ax2, y_axis='log')
    fig.colorbar(img, ax=ax2)

    ax3.set_title("Amplitude of percussive")
    ax3.plot(librosa.amplitude_to_db(np.sum(np.abs(D_percussive),axis=0)))
    
    ax4.set_title("Estimated pitch (pyin)")
    b = librosa.istft(D_harmonic, n_fft=fs//rHz, hop_length=int(fs*rs) )
    c = librosa.istft(D_percussive, n_fft=fs//rHz, hop_length=int(fs*rs) )
    f0, _, voiced_prob = librosa.pyin(b, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        sr = fs, resolution=1, max_transition_rate=2,)
    ax4.plot(f0)
    '''
    ax4.set_title("Estimated pitch (piptrack)")
    pitches, magnitudes = librosa.piptrack(S=D_harmonic, fmin=100, fmax=4000, )
    ax4.plot(pitches)
    '''

    if outfile is None:
        plt.show()
    else:
        fig.savefig(outfile)
        _, outfile = path.split(infile)
        outfile,_,_ = outfile.partition('.')
        outfile = path.join('wav_out',outfile+'.wav')
        newsound = np.stack([b,c]).T
        sf.write(outfile, newsound, fs, subtype='FLOAT')

d = 'wav'
e = 'img'
a = listdir('wav')
for x in a:
    infile = path.join(d,x)
    outfile = path.join(e,x.partition('.')[0]+'_librosa.png')
    print(x)
    spectrum(infile,outfile)

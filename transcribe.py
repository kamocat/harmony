import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def from_file(infile):
    a, fs  = sf.read(infile)
    a = librosa.to_mono(a.T)
    rHz = 10 #Frequency resolution in Hz
    rs = 0.01 #Temporal resolution in seconds
    
    # Perform FFT
    D = librosa.stft(a, n_fft=fs//rHz, hop_length=int(fs*rs) )
    freqs = librosa.fft_frequencies(sr=fs, n_fft=fs//rHz)
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=8)
    
    fig, (ax3) = plt.subplots(1)
    
    A_percussive = librosa.amplitude_to_db(np.sum(np.abs(D_percussive),axis=0))
    
    peaks,properties = find_peaks(A_percussive, height=45 )
    print(f'Attacks found at {peaks}')
    salience = librosa.salience(np.abs(D_harmonic), freqs=freqs, 
                                harmonics=list(range(1,8)))
    salience = np.nan_to_num(salience)
    
    ax3.set_title("Pitch")
    pitch = [freqs[x] for x in np.argmax(salience, axis=0)]
    peaks2 = np.append(peaks[1:], len(A_percussive)-1)
    median_pitch = [np.median(pitch[x:y]) for x,y in zip(peaks, peaks2)]
    print(f'You played {librosa.hz_to_note(median_pitch)}')
    
    melody = librosa.hz_to_midi(median_pitch)
    melody = [int(round(x)) for x in melody]
    lengths = librosa.frames_to_time(peaks2-peaks, sr=fs, hop_length=int(fs*rs))
    return melody, lengths

import scipy.signal as sig
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

def from_file(fname, debug=False):
    fs, a = read(fname)
    print(f'{fname} dimensions: {a.shape}')
    N = len(a)
    rHz = 10 #Frequency resolution in Hz
    rs = 0.01 #Temporal resolution in seconds

    # Set up the FFT
    SFT = sig.ShortTimeFFT(sig.windows.hamming(fs//rHz, sym=False),
                                int(fs*rs), fs, scale_to='magnitude')

    # Perform FFT
    Sx = SFT.stft(a)
    mag = np.abs(Sx)

    min_note = 0
    max_note = 100
    max_pitch = mag.shape[0]
    tonotes= np.zeros((max_pitch, max_note-min_note))
    lo = 2**(-1/24)
    hi = 2**(1/24)

    for i in range(min_note, max_note ):
        # Convert MIDI note to center pitch
        pitch = 440 / rHz * (2**((i-69)/12)) 
        n = 0
        for amp in [1, 0.5, 0.3, 0.25]:
            n += pitch 
            # Generate high and low range for each overtone
            lo2 = int(floor(lo * n))
            if lo2 >= max_pitch-1:
                break #We ran out of spectrum
            hi2 = int(min(ceil(hi * n + 1),max_pitch))
            # Update the transform matrix
            tonotes[lo2:hi2,i-min_note]=amp

    envelope = np.max(mag, axis=0)
    keys = np.argmax(np.matmul(mag.T, tonotes), axis=1)

    # Get note start times
#    attack = sig.savgol_filter(envelope, 10, 2, deriv=1, mode='nearest')
    thresh = np.max(envelope) * 0.1
    times = sig.find_peaks(envelope, height=thresh)[0]
    melody = keys[times]
    
    if debug:
        # Plot the data
        fig, [[ax1,ax2],[ax3,ax4]]= plt.subplots(2,2)
        #ax4.plot(attack, color='tab:green')
        ax3.step(times, melody)
        color = 'tab:blue'
        ax2.set_ylabel('Amplitude')
        ax2.plot(envelope, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        color = 'tab:red'
        ax1.set_xlabel(f'Time ({rs} seconds)')
        ax1.set_ylabel('MIDI key')
        ax1.plot(keys, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.show()

    times = np.append(times, len(envelope))
    durations = rs * (times[1:] - times[:-1])

    return melody, durations

if __name__ == '__main__':
    from_file('wav/sample3.wav', debug=True)


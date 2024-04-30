import scipy.signal as sig
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt

fs, a = read('sample1.wav')
print(a.shape)
N = len(a)
rHz = 10 #Frequency resolution in Hz
rs = 0.01 #Temporal resolution in seconds

# Set up the FFT
SFT = sig.ShortTimeFFT(sig.windows.hamming(fs//rHz, sym=False),
                            int(fs*rs), fs, scale_to='magnitude')

# Perform FFT
Sx = SFT.stft(a)

print(Sx.shape)

mag = np.abs(Sx)
peaks = mag.argmax(axis=0) * rHz
amplitude = mag.max(axis=0)
amplitude = np.log10(amplitude)*20
t = [x*rs for x in range(len(peaks))]
fig,(ax1,ax2) = plt.subplots(2)
ax1.set_title('Loudest frequency vs time')
ax1.set(xlabel="Time (seconds)", ylabel="Frequency (Hz)")
ax1.plot(t,peaks)
ax2.set_title('Relative amplitude of that frequency')
ax2.set(xlabel="Time (seconds)", ylabel="Amplitude (dB)")
ax2.plot(t,amplitude)
fig.tight_layout()
plt.show()

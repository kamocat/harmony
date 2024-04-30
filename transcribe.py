import scipy.signal as sig
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt

fs, a = read('sample1.wav')
print(a.shape)
N = len(a)

# Set up the FFT
SFT = sig.ShortTimeFFT(sig.windows.hamming(fs//10, sym=False),
                            fs//100, fs, scale_to='magnitude')

# Perform FFT
Sx = SFT.stft(a)

print(Sx.shape)

peaks = np.abs(Sx).argmax(axis=0)
plt.plot(peaks)
plt.show()

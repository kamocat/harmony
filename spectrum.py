import sounddevice as sd
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

fig1, ax1 = plt.subplots(figsize=(6.,4.))
t_lo,  t_hi = SFT.extent(N)[:2]
ax1.set_title("Waterfall")
ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
               rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " +
               rf"$\Delta f = {SFT.delta_f:g}\,$Hz)",
        xlim=(t_lo, t_hi))

im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                 extent=SFT.extent(N), cmap='viridis')
fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

# Shade areas where window slices stick out to the side:
for t0_, t1_ in [(t_lo, SFT.lower_border_end[0] * SFT.T),
                 (SFT.upper_border_begin(N)[0] * SFT.T, t_hi)]:
    ax1.axvspan(t0_, t1_, color='w', linewidth=0, alpha=.2)
for t_ in [0, N * SFT.T]:  # mark signal borders with vertical line:
    ax1.axvline(t_, color='y', linestyle='--', alpha=0.5)
fig1.tight_layout()
plt.show()

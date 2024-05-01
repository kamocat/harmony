import scipy.signal as sig
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from os import path

def spectrum(fname):
    fs, a = read(fname)
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

    fig1, (ax1,ax2) = plt.subplots(2)
    t_lo,  t_hi = SFT.extent(N)[:2]
    ax1.set_title("Spectral intensity plot")
    ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
                   rf"$\Delta t = {SFT.delta_t:g}\,$s)",
            ylabel=f"Freq. $f$ in Hz (" +
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

    # Add plots to track frequency and amplitude
    mag = np.abs(Sx)
    peaks = mag.argmax(axis=0) * rHz
    amplitude = mag.max(axis=0)
    amplitude = np.log10(amplitude)*20
    t = [x*rs for x in range(len(peaks))]
    color = 'tab:red'
    ax2.set_title('Loudest frequency vs time')
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Frequency (Hz)", color=color)
    ax2.plot(t,peaks, color=color)
    ax3 = ax2.twinx()
    color = 'tab:blue'
    ax3.set_ylabel("Amplitude (dB)", color=color)
    ax3.plot(t,amplitude, color=color)
    fig1.tight_layout()
    fig1.savefig(fname.partition('.')[0]+'.png')


spectrum('sample1.wav')

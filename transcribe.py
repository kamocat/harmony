import scipy.signal as sig
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

fs, a = read('wav/sample1.wav')
print(a.shape)
N = len(a)
rHz = 1 #Frequency resolution in Hz
rs = 0.01 #Temporal resolution in seconds

# Set up the FFT
SFT = sig.ShortTimeFFT(sig.windows.hamming(fs//rHz, sym=False),
                            int(fs*rs), fs, scale_to='magnitude')

# Perform FFT
Sx = SFT.stft(a)

print(Sx.shape)

mag = np.abs(Sx)

min_note = 21 #Lowest piano note A0
max_note = 94 #A6 + 1
max_pitch = mag.shape[1]
tonotes= np.zeros((max_pitch, max_note-min_note))
lo = 2**(1/24)
hi = 2**(-1/24)

for i in range(min_note, max_note ):
    # Convert MIDI note to center pitch
    pitch = 440 / rHz * (2**((i-69)/12)) 
    print(f'key {i-min_note} pitch {pitch}Hz')
    n = 0
    for amp in [1,.5,.3,.25]:
        n += pitch 
        # Generate high and low range for each overtone
        lo2 = floor(lo * n)
        if lo2 >= max_pitch-1:
            #print(f'Note {i} pitch {pitch*rHz} too high at amp {amp}')
            break #We ran out of spectrum
        hi2 = min(ceil(hi * n),max_pitch-1)
        # Update the transform matrix
        tonotes[lo2:hi2,i-min_note]=amp
print(np.sum(tonotes,axis=0))


    
notes = np.matmul(mag, tonotes)
print(notes.shape)

plt.imshow(tonotes.T, cmap='gray', vmin='0', vmax='1')
plt.show()

if 0:
    fig1, ax1 = plt.subplots()
    t_lo,  t_hi = SFT.extent(N)[:2]
    ax1.set_title("Note intensity plot")
    ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
                   rf"$\Delta t = {SFT.delta_t:g}\,$s)",
            ylabel=f"Keys of a piano",
            xlim=(t_lo, t_hi))

    im1 = ax1.imshow(notes, origin='lower', aspect='auto',
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

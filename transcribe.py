import scipy.signal as sig
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil

fs, a = read('wav/sample8.wav')
print(f'File dimensions: {a.shape}')
N = len(a)
rHz = 1 #Frequency resolution in Hz
rs = 0.01 #Temporal resolution in seconds

# Set up the FFT
SFT = sig.ShortTimeFFT(sig.windows.hamming(fs//rHz, sym=False),
                            int(fs*rs), fs, scale_to='magnitude')

# Perform FFT
Sx = SFT.stft(a)

print(f'FFT dimensions: {Sx.shape}')

mag = np.abs(Sx)

min_note = 0
max_note = 127
max_pitch = mag.shape[0]
tonotes= np.zeros((max_pitch, max_note-min_note))
lo = 2**(-1/24)
hi = 2**(1/24)

for i in range(min_note, max_note ):
    # Convert MIDI note to center pitch
    pitch = 440 / rHz * (2**((i-69)/12)) 
    #print(f'key {i-min_note} pitch {pitch:.0f} Hz')
    n = 0
    for amp in [1,.3,.1,.03]:
        n += pitch 
        # Generate high and low range for each overtone
        lo2 = int(floor(lo * n))
        if lo2 >= max_pitch-1:
            #print(f'Note {i} pitch {pitch*rHz} too high at amp {amp}')
            break #We ran out of spectrum
        hi2 = int(min(ceil(hi * n + 1),max_pitch))
        # Update the transform matrix
        tonotes[lo2:hi2,i-min_note]=amp
print(f'Transform weights: {np.sum(tonotes,axis=0)}')


    
notes = np.matmul(mag.T, tonotes)
print(f'Notes matrix shape: {notes.shape}')

#plt.imshow(notes.T, cmap='gray', vmin='0', vmax=notes.max())
#plt.show()

if 1:
    fig1, ax1 = plt.subplots()
    t_lo,  t_hi = SFT.extent(N)[:2]
    ax1.set_title("Note intensity plot")
    ax1.set(xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " +
                   rf"$\Delta t = {SFT.delta_t:g}\,$s)",
            ylabel=f"Midi note",
            xlim=(t_lo, t_hi))

    im1 = ax1.imshow(notes.T, origin='lower', aspect='auto',
                     extent=SFT.extent(N), cmap='viridis')
    fig1.colorbar(im1, label="Magnitude $|S_x(t, f)|$")

    fig1.tight_layout()
    plt.show()

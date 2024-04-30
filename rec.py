import sounddevice as sd
from scipy.io.wavfile import write

fs = 8000  # Sample rate
seconds = 5  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write('sample.wav', fs, myrecording)

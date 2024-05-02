# From https://github.com/spatialaudio/python-sounddevice/blob/0.4.6/examples/rec_unlimited.py
import queue
from os import path, listdir
import sys

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

def nname(fname='sample.wav', mydir='wav'):
    '''Gets a new (unused) filename, numerically incremented'''
    p,_,s = fname.partition('.')
    a = listdir(mydir)
    n = 0
    for x in a:
        b = x.partition(p)[2]
        c = b.partition('.')[0]
        if c.isnumeric():
            n = max(n,int(c))
    fname = path.join(mydir, f'{p}{n+1}.{s}')
    return fname

q = queue.Queue()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


try:

    samplerate = 44100
    channels = 1
    # Make sure the file is opened before recording anything:
    f = nname()
    with sf.SoundFile(f, mode='x', samplerate=samplerate,
                      channels=channels) as file:
        with sd.InputStream(samplerate=samplerate, 
                            channels=channels, callback=callback):
            print('#' * 80)
            print('press Ctrl+C to stop the recording')
            print('#' * 80)
            while True:
                file.write(q.get())

except KeyboardInterrupt:
    print('\nRecording finished: ' + f)
except Exception as e:
    print(e, file=sys.stederr)

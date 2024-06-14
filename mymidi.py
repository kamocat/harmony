import mido
import transcribe

mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
track.append(mido.Message('program_change', program=0, time=0))

melody, lengths = transcribe.from_file('wav/sample8.wav')

for m, l in zip(melody, lengths):
    track.append(mido.Message('note_on',note=m, time=int(480 * l), velocity=127))
    #track.append(mido.Message('note_off',note=m, time=1, ))


mid.save('new_song.mid')

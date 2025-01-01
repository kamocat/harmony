import mido
import transcribe

mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
track.append(mido.Message('program_change', program=1, time=0))

melody, lengths = transcribe.from_file('wav/test_cegfeddc.wav')

PQQN = 500
tempo = mido.bpm2tempo(120)

for m, l in zip(melody, lengths):
    t = int(mido.second2tick(l, PQQN, tempo))
    #Attack time should be short
    track.append(mido.Message('note_on',note=m, time=1, velocity=127))
    #Release time is the length of the note
    track.append(mido.Message('note_off',note=m, time=t, ))


mid.save('new_song.mid')

import mido

mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
track.append(mido.Message('program_change', program=8, time=0))

for i in range(60,73):
    track.append(mido.Message('note_on',note=i, time=240, velocity=127))
    track.append(mido.Message('note_off',note=i, time=1, ))


mid.save('new_song.mid')

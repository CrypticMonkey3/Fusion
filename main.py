import tensorflow as tf
from music21 import *
import numpy as np
import os


class Fusion:
    def __init__(self):
        self.__running = True

    # The preprocessing code was taken from this page:
    # https://thehackerstuff.com/getting-started-with-ai-music-generation-exploring-the-possibilities/
    def __preprocess_midi(self, file_dir: str):
        """

        :param str file_dir: The directory of the file we want to use.
        :return:
        """
        midi_data = converter.parse(file_dir)  # loads MIDI file and makes it a Music21 stream object

        pitches, durations, velocities = [], [], []
        for note in midi_data.notes:
            pitches.append(note.pitch.midi)
            durations.append(note.duration.quarterLength)
            velocities.append(note.velocity)

    def __process(self, file_dir: str) -> None:
        """
        Contains the sequence of methods that Fusion will execute.
        :return: None
        """
        self.__preprocess_midi(file_dir)

    def run(self) -> None:
        """
        Begins running the Fusion music generator
        :return: None
        """
        while self.__running:
            file_index = int(input("Type index of midi file to be used:\n>"))
            self.__process(f"Music-Dataset/{os.listdir('Music-Dataset')[file_index]}")


x = converter.parse("Music-Dataset/Cymatics - Lofi MIDI 1 - C Maj.mid")
print(x)
x = x.makeMeasures()  # splits original music into measures
print(x)
x = x.flat  # combines the measures to a single stream
print(x)
x = x.getElementsByClass("Note")  # selects all the notes from the stream
print(x)


f = Fusion()
f.run()



import tensorflow as tf
from music21 import *
import numpy as np
import os


class Fusion:
    def __init__(self):
        self.__running = True

    def __preprocess_midi(self, file_dir: str):
        """

        :param str file_dir: The directory of the file we want to use.
        :return:
        """
        pitches, durations, velocities = [], [], []

        midi_data = converter.parse(file_dir)  # loads MIDI file and makes it a Music21 stream object

        notes_to_parse = midi_data.flatten().notes  # assume file has notes in a flat structure
        instruments = instrument.partitionByInstrument(midi_data)
        print(instruments)
        if instruments is not None:  # if the file has instrumental parts
            # creates a RecursiveIterator object from it.
            notes_to_parse = instruments.recurse()

            print(instruments.recurse())

        print(notes_to_parse)
        for element in notes_to_parse:  # goes through each note in the midi file
            match type(element):
                case note.Note:
                    pitches.append(str(element.pitch))
                    durations.append(str(element.duration.quarterLength))

                case chord.Chord:
                    print(".".join(str(_) for _ in element.normalOrder))

        # extract certain attributes from midi data
        # pitches, durations, velocities = [], [], []
        # for instr_note in midi_data.flat.notes:
        #     pitches.append(instr_note.pitch.midi)
        #     durations.append(instr_note.duration.quarterLength)
        #     velocities.append(instr_note.velocity)
        #     print("1")
        #
        # print(pitches)
        # print(durations)
        # print(velocities)
        #
        # # convert pitch, duration, and velocity to numpy arrays: a more efficient data structure for numerical operation
        # pitches = np.array(pitches)
        # durations = np.array(durations)
        # velocities = np.array(velocities)
        #
        # print(pitches)
        # print(durations)
        # print(velocities)

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


f = Fusion()
f.run()

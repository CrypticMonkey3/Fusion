from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from music21 import *
from typing import *
import numpy as np
import os


class Fusion:
    def __init__(self):
        self.__running = True

    @staticmethod
    def __preprocess_midi(file_dir: str) -> np.stack:
        """
        Take a midi file and get relevant features: notes, and duration
        :param str file_dir: The directory of the file we want to use.
        :return: np array, containing data for each feature.
        """
        durations, notes = [], []

        midi_data = converter.parse(file_dir)  # loads MIDI file and makes it a Music21 stream object

        notes_to_parse = midi_data.flatten().notes  # get a Stream of all the notes in the MIDI files
        instruments = instrument.partitionByInstrument(midi_data)  # make a Part object for each unique instrument.

        if instruments:  # if the file has instrumental parts
            # creates a RecursiveIterator object from it.
            notes_to_parse = instruments.recurse()

        for element in notes_to_parse:  # goes through each note in the midi file
            match type(element):  # depending on element type, we extract certain attributes
                case note.Note:
                    notes.append(str(element.pitch))
                    durations.append(str(element.duration.quarterLength))

                case chord.Chord:  # a chord is a sequence of notes
                    notes.append(".".join(str(_) for _ in element.normalOrder))
                    durations.append(str(element.duration.quarterLength))

        # return np array that has the following structure
        # [   [   note    ]   [   duration    ]   ]
        return np.stack((notes, durations)).T

    @staticmethod
    def __prepare_sequence(notes: np.stack, sequence_length) -> Tuple[List[np.array], List[np.array], Any]:
        """
        Create an input sequence and their corresponding outputs, and use One-Hot-Encoding.
        :param np.stack notes: A numpy stack containing notes/chords and relevant features associated with them.
        :param int sequence_length: The length of our note sequences.
        :return: A list containing One-Hot-Encoded sequences, and another list containing the next note/chord after each sequence.
        """
        network_input, network_output = [], []
        onehot_encoder = OneHotEncoder(sparse_output=False)  # sparse=False because sparse encoding is not suitable for keras
        print(notes)

        onehot_notes = onehot_encoder.fit_transform(notes)  # One-Hot-Encode our notes

        # --- One-Hot-Encode each column in the notes stack --- #
        # label_encoder = LabelEncoder()
        # for column in notes.T:  # transpose notes, so we get to iterate through each column
        #     int_encode_col = label_encoder.fit_transform(column)  # gives each feature of that column a numerical label
        #     print(int_encode_col)
        #     int_encode_col = int_encode_col.reshape(len(int_encode_col), 1)  # reshape int_encode_col to len(int_encode_col)x1 matrix
        #     print(int_encode_col)
        #     onehot_encode_col = onehot_encoder.fit_transform(int_encode_col)
        #     print(onehot_encode_col)

        # --- Create input sequences and each sequences corresponding output --- #
        for i in range(len(onehot_notes) - sequence_length):  # loops until all sequences and corresponding outputs are retrieved.
            sequence_head = sequence_length + i  # the end index of a given sequence
            network_input.append(onehot_notes[i:sequence_head, :])  # append a sequence to the network inputs
            network_output.append(onehot_notes[sequence_head, :])  # append the next note/chord after the current sequence to output.

        print(network_input)
        print(network_output)

        return network_input, network_output, onehot_notes

    @staticmethod
    def __build_model(input_shape, num_pitches: int) -> keras.Model:
        """
        Create a Sequential model containing various hidden layers and activation layers best suited for the problem
        :param input_shape: The shape of our training data. This ensures each tensor (flow between layers) have the correct dimensions
        :param int num_pitches: The number of unique notes/chords.
        :return:
        """
        # --- Architecture of Model --- #
        # inputs = keras.Input(input_shape)
        # dropout1 = keras.layers.Dropout(0.2)(inputs)  # randomly sets input units to 0 every .2s to reduce overfitting
        # lstm1 = keras.layers.LSTM(512, return_sequences=True)(dropout1)  # return_sequences=True, for outputting the last output
        # dense1 = keras.layers.Dense(256)(lstm1)  # applies some operations to change the dimensions of the vector. We're expecting 256 output size
        # dense2 = keras.layers.Dense(256)(dense1)
        # lstm2 = keras.layers.LSTM(512, return_sequences=True)(dense2)  # units represent the number of neurons
        # dense3 = keras.layers.Dense(256)(lstm2)
        # lstm3 = keras.layers.LSTM(512)(dense3)  # return_sequences=False, so we return the full sequence
        # dense4 = keras.layers.Dense(num_pitches)(lstm3)
        # activation1 = keras.layers.Activation("softmax")(dense4)  # converts vector to probability distribution

        # model = keras.Model(inputs=inputs, outputs=activation1)

        model = keras.Sequential(
            [
                keras.layers.Input(input_shape),
                keras.layers.Dropout(0.2),  # randomly sets input units to 0 every .2s to reduce overfitting
                keras.layers.LSTM(512, return_sequences=True),  # return_sequences=True, for outputting the last output
                keras.layers.Dense(256),  # applies some operations to change the dimensions of the vector. We're expecting 256 output size
                keras.layers.Dense(256),
                keras.layers.LSTM(512, return_sequences=True),  # units represent the number of neurons
                keras.layers.Dense(256),
                keras.layers.LSTM(512),  # return_sequences=False, so we return the full sequence
                keras.layers.Dense(num_pitches),
                keras.layers.Activation("softmax")  # converts vector to probability distribution
            ]
        )

        # builds the model architecture: num of hidden layers and activation layers
        model.compile("Adam", "categorical_crossentropy", ["accuracy"])

        return model

    @staticmethod
    def __train_model(model, epochs: int, network_input, network_output):
        """

        :param keras.Sequential model: The model being trained using network_input and network_output
        :param int epochs: The number of epochs to have during training.
        :param network_input:
        :param network_output:
        :return:
        """
        # Checkpoints are used to save your model in case your system crashes or the code was interrupted whilst training, so you don't have to start
        # from scratch. Checkpoints don't contain any description of the computation defined by the model.
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            "checkpoint.weights.h5",  # ends in weights.h5 when save_weights_only=True, otherwise .keras
            "val_accuracy",
            0b1,
            True,
            True,  # faster
            "max"
        )
        print(checkpoint_callback)

        # train and save the best model weights at the end of every epoch
        model.fit(
            network_input,
            network_output,
            0b100000,
            epochs,
            "auto",
            [checkpoint_callback],
        )

        return model

    def __process(self, file_dir: str) -> None:
        """
        Contains the sequence of methods that Fusion will execute.
        :return: None
        """
        notes = self.__preprocess_midi(file_dir)
        network_inp, network_out, x = self.__prepare_sequence(notes, 0b11)

        # print(np.array(network_inp))
        # print(np.array(network_inp).shape)

        model = self.__build_model(network_inp[0].shape, len(set(notes[:, 0])))
        model = self.__train_model(model, 0b1100100, network_inp, network_out)

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

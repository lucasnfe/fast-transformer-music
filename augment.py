#
# Augment MIDI data with transposition and time strech to improve
# neural networks performance.
#
# Author: Lucas N. Ferreira - lucasnfe@gmail.com
#
# Transoposition: https://en.wikipedia.org/wiki/Transposition_(music)
#

import os
import copy
import argparse
import pretty_midi

MIDI_EXTENSIONS = [".mid", ".midi"]

def has_midi_files(dir_path):
    files = os.listdir(dir_path)

    for f in files:
        filename, extension = os.path.splitext(f)
        if extension in MIDI_EXTENSIONS:
            return True

    return False

def transpose(in_file_path, intervals, out_file_path):
    for i in intervals:
        # Make a copy of the original midi
        tranposed_mid = pretty_midi.PrettyMIDI(midi_file=in_file_path)

        # Transpose all pitched notes
        for instrument in tranposed_mid.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    note.pitch += i

        if i < 0:
            # Interval is a flat (b)
            interval_name = "_b_" + str(abs(i))
        elif i == 0:
            interval_name = "_original"
        elif i > 0:
            # Interval is a sharp (#)
            interval_name = "_#_" + str(abs(i))

        # Save transposition
        filename, _ = os.path.splitext(out_file_path)
        tranposed_mid.write(filename + interval_name + ".mid")

def strech(in_file_path, stretch_factors, out_file_path):
    for sf in stretch_factors:
        # Make a copy of the original midi
        streched_mid = pretty_midi.PrettyMIDI(midi_file=in_file_path)

        # Strech time of all notes
        for instrument in streched_mid.instruments:
            for note in instrument.notes:
                # stretch note start time
                note.start *= sf

                # stretch note end time
                note.end *= sf

        if sf < 0:
            # This is a slower version
            strech_name = "_slow_" + str(int(sf * 1000))
        elif sf == 0:
            # This is the original version
            strech_name = "_original"
        elif sf > 0:
            # This is a faster version
            strech_name = "_fast_" + str(int(sf * 1000))

        # Save streched version
        filename, _ = os.path.splitext(out_file_path)
        streched_mid.write(filename + strech_name + ".mid")

def augment_midi_data(midi_path, transpose_intervals, stretch_factors):
    # Get dir content outside for to avoid infinite loop
    dir_content = os.walk(midi_path)

    for dir, _ , files in dir_content:
        # Create augmented folder
        if has_midi_files(dir):
            # Only create augmented versions of directories with midi files
            dir_augmented = dir + "_augmented"
            os.makedirs(dir + "_augmented", exist_ok=True)

            for i,f in enumerate(files):
                # Check that file has valid midi extension
                filename, extension = os.path.splitext(f)

                if extension.lower() in MIDI_EXTENSIONS:
                    in_file_path = os.path.join(dir, f)
                    out_file_path = os.path.join(dir_augmented, f)

                    # Transpose midi
                    transpose(in_file_path, transpose_intervals, out_file_path)

                    # Strech midi
                    strech(in_file_path, stretch_factors, out_file_path)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='augment.py')
    parser.add_argument('--midi', type=str, required=True, help="Path to midi data.")
    opt = parser.parse_args()

    # Range of transposition intervals (in semi-tones)
    transpose_intervals = range(-4, 5)

    # Range of strech factors
    stretch_factors = [0.95, 0.975, 1, 1.025, 1.05]

    augment_midi_data(opt.midi, transpose_intervals, stretch_factors)

import os
import wave
import struct
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Dataset Directories
input_path = '' # input directory path here
output_path = '' # output directory path here
os.makedirs(output_path, exist_ok=True)

# convert .wav to .jpg for each file in audio dataset
for file_name in os.listdir(input_path):
    if file_name.endswith('.wav'):

        # Read .WAV Files
        input_file = os.path.join(input_path, file_name)
        file = wave.open(input_file) # open .wav audio file
        frames = file.readframes(-1) # read all audio frames in the file

        # unpack audio frames into tuple
        samples = struct.unpack('h' * file.getnframes(), frames) 
        # format char 'h' - interpret bytes as 16-bit signed integers 

        time = [] # time values for each frame
        framerate = file.getframerate() 
        for i in range(len(samples)):
            time.append(float(i) / framerate)

        # Noise Reduction
        reduced = savgol_filter(samples, 501, 2)    # reduce the middle number (501) 
                                                    # for less noise reduction

        # Plot waveform amplitude against time
        plt.figure()
        plt.plot(time, reduced, color='black')    # high frequency noise removed
        plt.axis('off')

        # Export JPGs
        output_graph = os.path.join(output_path, file_name.replace('.wav', 'nonoise.jpg'))
        plt.savefig(output_graph)
        plt.close()

        print(f'saved graph for {file_name} to {output_graph}')

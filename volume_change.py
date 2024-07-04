import wave
import audioop
import os

factor = 1.5 # change volume by this factor

input_path = '' # input directory path here
output_path = '' # output directory path here
os.makedirs(output_path, exist_ok=True)

for file_name in os.listdir(input_path):
    if file_name.endswith('.wav'):

        # Read .WAV Files
        input_file = os.path.join(input_path, file_name)
        output_file = os.path.join(output_path, file_name)
        with wave.open(input_file, 'rb') as wav:
            param = wav.getparams()
            with wave.open(output_file, 'wb') as audio:
                audio.setparams(param)
                frames = wav.readframes(param.nframes)
                audio.writeframesraw(audioop.mul(frames, param.sampwidth, factor))

        print(f'saved graph for {file_name} to {output_file}')

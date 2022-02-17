import os
import librosa

#from pydub import AudioSegment
#from pyannote.audio.utils.signal import Binarize, Peak
#from pyannote.core import Segment, notebook, SlidingWindowFeature, timeline, Timeline

#Specify path to codec exe used by pydub
#AudioSegment.converter='C:\\Users\\MrBrito\\Documents\\ffmpeg\\ffmpeg-2021-04-20-git-718e03e5f2-full_build\\ffmpeg-2021-04-20-git-718e03e5f2-full_build\\bin\\ffmpeg.exe'


def downsamp_audio(directory, rate=16000):
#Takes a directory of .wav files and downsamples them to rate e.g. rate=16000 means downsampled to 16kHz
    print("hello")
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            clear_output(wait=True)
            new_rate = rate
            resamp, rate = librosa.load(os.path.join(raw, filename), sr=new_rate)
            sf.write(os.path.join(directory, filename), resamp, new_rate)
            print("processing" + str(filename))
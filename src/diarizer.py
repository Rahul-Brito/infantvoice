import os
import sys
from pyannote.audio import Pipeline
from pyannote.core import json

director = sys.argv[1]
out_dir = sys.argv[2]

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

for file in os.listdir(directory):
    diar = pipeline(os.path.join(directory,file))
    json.dump_to(diar, os.path.join(out_dir, os.path.splitext(file)[0]+'.json'))
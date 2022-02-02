import os
import numpy as np
import pandas as pd
from IPython.display import clear_output
import torch
from pyannote.audio.utils.signal import Binarize, Peak
from pyannote.core import Segment, notebook, SlidingWindowFeature, timeline, Timeline

# speech activity detection model trained on AMI training set
sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')
# speaker change detection model trained on AMI training set
scd = torch.hub.load('pyannote/pyannote-audio', 'scd_ami')
# speaker embedding model trained on AMI training set
#emb = torch.hub.load('pyannote/pyannote-audio', 'emb_ami')
#emb = torch.hub.load('pyannote/pyannote-audio', 'emb')
emb = torch.hub.load('pyannote/pyannote-audio', 'emb_voxceleb')

def pyannote_run_directory(directory, save_dir, save_name, save=False):

    #extracts embeddings from .wav files in a folder
    #does speaker activity and change detection for long sentences, but just embedding extraction for concatenated hellos
    #using pyannote audio using tutorial: https://github.com/pyannote/pyannote-audio/tree/master/tutorials/pretrained/model
    
    #all_embs, = [], [], []

    print("Started")#to keep track of progress when running multiple participants
    
    #finds all .wav files in target directory, will extract embeddings from each
    all_embs = []
    part_id = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".wav"): 
            
            #clears output then prints only current sample being operated on
            clear_output(wait=True)
            print("Processing" + str(filename))
            
            #uri format required for pyannote audtio
            one_file = {'uri': 'filename', 'audio': os.path.join(directory, filename)}
            
            #extract embeddings from one wav file
            #note that this is modifying from pyannote example
            emb_from_sample = pyannote_extract_embs(one_file)
            
            part_id.append([filename[0:2]]*emb_from_sample.shape[0])
            all_embs.append(emb_from_sample)

    all_embs = pd.DataFrame(np.vstack(all_embs))
    all_embs['part_id'] = np.hstack(part_id)
    
    print("Done")
    
    if save:
        all_embs.to_csv(os.path.join(save_dir, save_name))

    return all_embs




def pyannote_extract_embs(one_file):
    
    emb_from_sample = []
    
    # obtain raw embeddings (as `pyannote.core.SlidingWindowFeature` instance)
    embeddings = emb(one_file)

    # obtain raw SAD scores (as `pyannote.core.SlidingWindowFeature` instance), binarize raw SAD scores
    # NOTE: both onset/offset values were tuned on AMI dataset. you might need to use different values for better results
    sad_scores = sad(one_file)
    binarize = Binarize(offset=0.52, onset=0.52, log_scale=True, min_duration_off=0.1, min_duration_on=0.1)
    speech = binarize.apply(sad_scores, dimension=1)

    # obtain raw SCD scores (as `pyannote.core.SlidingWindowFeature` instance), detect peaks & return speaker homogeneous segments 
    # NOTE: both alpha/min_duration values were tuned on AMI dataset. you might need to use different values for better results.
    scd_scores = scd(one_file)
    peak = Peak(alpha=0.10, min_duration=0.10, log_scale=True)

    # speech turns are simply the intersection of SAD and SCD
    partition = peak.apply(scd_scores, dimension=1)
    speech_turns = partition.crop(speech)

    #We only work of long (> ts) speech turns
    t = 2
    long_turns = Timeline(segments=[s for s in speech_turns if s.duration > t])

    for segment in long_turns:
        inter = embeddings.crop(segment, 'strict')
        emb_from_sample.append(inter)
        #emb_from_sample.append(np.mean(inter, axis=0)
        #emb_from_sample = np.row_stack((emb_from_sample, np.mean(inter, axis=0)))
    emb_from_sample = np.vstack(emb_from_sample)
    return emb_from_sample


#def VFPpara_sid_creator(filename):
    
    #p = filename[filename.index('_')-2:filename.index('_')]
    #vtype = filename[filename.index('_')+1]
    #idn = os.path.splitext(filename)[0][-1]
    #if idn=='h':
    #    idn=''
    #stim_type.append(vtype+idn)
    
    #if 'Norm' in filename:
    #    sid_per_sample.append(p + 'n')
    #else:
    #    sid_per_sample.append(p + 'v')
    #
    
    #sid_per_sample, stim_type = [],[]
    
    #for i in range(0, inter.shape[0]):
    #    p = filename[filename.index('_')-2:filename.index('_')]
    #    vtype = filename[filename.index('_')+1]
    #    idn = os.path.splitext(filename)[0][-1]
    #    if idn=='h':
    #        idn=''
    #    stim_type.append(vtype+idn)
    #    if 'Norm' in filename:
    #        sid_per_sample.append(p + 'n')
    #    else:
    #        sid_per_sample.append(p + 'v')
    #return sid_per_sample,stim_type
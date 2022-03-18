import os
import numpy as np
import pandas as pd

from IPython.display import clear_output

import torch

from pyannote.audio import Inference
from pyannote.audio.utils.signal import Binarize, Peak
from pyannote.core import Segment, notebook, SlidingWindowFeature, timeline, Timeline

from skimage.measure import block_reduce

emb = Inference("pyannote/embedding", 
                      window="sliding",
                      duration=3.0, step=1.0)

def pyannote_extract_directory(wav_files, diar, save_dir, save_name, save=False):

    #extracts embeddings from .wav files in a folder
    #does speaker activity and change detection for long sentences, but just embedding extraction for concatenated hellos
    #using modified version of pyannote audio using tutorial: https://github.com/pyannote/pyannote-audio/tree/master/tutorials/pretrained/model
    
    #wav_files: which directory to look for .wav files
    #diar: timelines with diarized data
    #save_dir: which directory to save to if save enabled
    #save_name: name of .csv to save embeddings to
    #save: flag of whether to save embeddings to csv or not. Default is to NOT save and therefore save_dir and save_name can be blank paths
   
    print("Started")#to keep track of progress when running multiple participants
    

    ## finds all .wav files in target directory, will extract embeddings from each 
    all_embs = [] #init list for all embeddings
    part_id = [] #init list for all participant labels
    for filename in os.listdir(wav_files):
        if filename.endswith(".wav"): 
            
            #clears output then prints only current sample being operated on
            clear_output(wait=True)
            print("Processing" + str(filename))
            
            one_file = os.path.join(wav_files, filename)
            
            one_diar = diar[int(os.path.splitext(filename)[0])]
            
            emb_from_sample = pyannote_extract_embs(one_file, one_diar)#extract embeddings from one .wav, using diarization map to just get intended speaker
            
            part_id.append([os.path.splitext(filename)[0]]*emb_from_sample.shape[0]) #create list of the participant ID as long as the number of embeddings from sample
            
            all_embs.append(emb_from_sample) #append embeddings from each participant into one long list

    all_embs = pd.DataFrame(np.vstack(all_embs))#turn list of embeddings into pandas dataframe
    all_embs['part_id'] = np.hstack(part_id)#add participant label to the dataframe
    
    print("Done")
    
    #saves embeddings to a csv if the flag is true
    if save:
        all_embs.to_csv(os.path.join(save_dir, save_name))

    return all_embs




def pyannote_extract_embs(one_file, one_diar):
       
    #using pyannote audio pretrained model tutorial as is: https://github.com/pyannote/pyannote-audio/tree/master/tutorials/pretrained/model
    #Main change from tutorial - don't calculate the means of the embeddings extracted from each 500ms speech turn.
    
    #one_file: .wav file in the format the model expects
    #diar: diarization map with pyannote timeline to just extract based on intended speaker
    
    emb_from_sample = []
    
    # obtain raw embeddings (as `pyannote.core.SlidingWindowFeature` instance)
    embeddings = emb(one_file)

    #We only work of long (> ts) speech turns. Default is 2s from the tutorial.
    # DIAR only has speech turns from desired participant
    t = 2
    long_turns = Timeline(segments=[s for s in one_diar if s.duration > t])

    #for each long turn of >t seconds long, extract each 500ms segment of embeddings 
    for segment in long_turns:
        inter = embeddings.crop(segment, 'strict')
        emb_from_sample.append(inter)
        #Tutorial calculated the mean of all the embeddings from a 500ms segment, but we keep them all
        #emb_from_sample.append(np.mean(inter, axis=0)
    emb_from_sample = np.vstack(emb_from_sample) #vstack to get the list into numpy format with easy to understand #of embeddings X embedding values structure
    return emb_from_sample



def load_embs(loc):
    
    #loads embeddings that were already extracted by taking in the location of a .csv file
    all_embs=pd.read_csv(loc, index_col=0)
    return all_embs



def resample_data(all_embs, factor, dropNA = True):
    
    #resamples the pandas dataframe containing all embeddings from all participants
    
    #all_embs: all embeddings from all participants
    #factor: factor to resample by. Overlap is dropped
    #dropNA: flag if drops NaN from embedding data or not. Default is true

    if dropNA:
        all_embs = all_embs.dropna(inplace = False)
    
    #shuffle the embeddings within each participant (shuffly by row)
    #moves embeddings from segements a particular speech turn away from each other
    #pd.groupby: Create an object separated by embeddings of each participant
    #pd.sample: Return a random sample of items from an axis of object (this shuffles)
    #### frac: Fraction of axis items to return
    #pd.reset_index: Reset the index so that the rows are numbered sequentially instead of by their old pre-shuffled index
    all_embs = all_embs.groupby('part_id').sample(frac=1).reset_index(drop=True)
    
    
    grp = all_embs.groupby('part_id') #creates an object separated by embeddings of each participant
    
    emb_down = []
    
    #find the mean of every <factor> set of embeddings within a participant, drop the carryover
    #after the shuffle, we are finding the means of shuffles from within a participant but from different segments of the audio
    for name, embeddings in grp:
            
            #downsamples by "factor". Block_reduce expected np not pd hence the conversion
            downsamp = block_reduce(embeddings.to_numpy(dtype='float32'), block_size=(factor, 1), func=np.mean, cval=np.nan)

            emb_down.append(downsamp)#append each participants resampled embeddings
    emb_down = pd.DataFrame(np.vstack(emb_down), columns=all_embs.columns).dropna().reset_index(drop=True)#remake as pd, drop NA rows(carryover rows), reset index
    return emb_down


def embedding_averager(emb):

    #finds the per-participant average embeddings (1x512 features)
    grp = emb.groupby('part_id')

    emb_a = []

    for name, embeddings in grp:
        #find the mean of all embeddings. Axis=0 means mean along the index because i want the mean of all 512 features
        emb_a.append(embeddings.mean(axis=0))
    emb_a = pd.DataFrame(emb_a)   
    return emb_a

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
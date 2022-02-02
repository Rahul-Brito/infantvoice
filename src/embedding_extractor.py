import os
import numpy as np
import pandas as pd
from IPython.display import clear_output
import torch
from pyannote.audio.utils.signal import Binarize, Peak
from pyannote.core import Segment, notebook, SlidingWindowFeature, timeline, Timeline




def pyannote_extract_directory(emb, directory, save_dir, save_name, save=False):

    #extracts embeddings from .wav files in a folder
    #does speaker activity and change detection for long sentences, but just embedding extraction for concatenated hellos
    #using modified version of pyannote audio using tutorial: https://github.com/pyannote/pyannote-audio/tree/master/tutorials/pretrained/model
    
    #emb: which pre-trained embedding model to use
    #directory: which directory to look for .wav files
    #save_dir: which directory to save to if save enabled
    #save_name: name of .csv to save embeddings to
    #save: flag of whether to save embeddings to csv or not. Default is to NOT save and therefore save_dir and save_name can be blank paths
   

    print("Started")#to keep track of progress when running multiple participants
    
    ### Choose which pre-trained embedding extractor to use
    # speaker embedding models trained on AMI training set
    
    if emb == 'emb_ami':
        emb = torch.hub.load('pyannote/pyannote-audio', 'emb_ami')
    elif emb == 'emb':
        emb = torch.hub.load('pyannote/pyannote-audio', 'emb')
    elif emb == 'emb_voxceleb':
        emb = torch.hub.load('pyannote/pyannote-audio', 'emb_voxceleb')
        
    ## finds all .wav files in target directory, will extract embeddings from each
    
    all_embs = [] #init list for all embeddings
    part_id = [] #init list for all participant labels
    for filename in os.listdir(directory):
        if filename.endswith(".wav"): 
            
            #clears output then prints only current sample being operated on
            clear_output(wait=True)
            print("Processing" + str(filename))
            
            one_file = {'uri': 'filename', 'audio': os.path.join(directory, filename)}#uri format required for pyannote audtio
            
            emb_from_sample = pyannote_extract_embs(emb, one_file)#extract embeddings from one .wav w/ modified pyannote example using the chosen embedding model
            
            part_id.append([filename[0:2]]*emb_from_sample.shape[0]) #create list of the participant ID that is as long as the number of embeddings from their sample
            
            all_embs.append(emb_from_sample) #append embeddings from each participant into one long list

    all_embs = pd.DataFrame(np.vstack(all_embs))#turn list of embeddings into pandas dataframe
    all_embs['part_id'] = np.hstack(part_id)#add participant label to the dataframe
    
    print("Done")
    
    #saves embeddings to a csv if the flag is true
    if save:
        all_embs.to_csv(os.path.join(save_dir, save_name))

    return all_embs




def pyannote_extract_embs(emb, one_file):
    
       
    #using pyannote audio pretrained model tutorial as is: https://github.com/pyannote/pyannote-audio/tree/master/tutorials/pretrained/model
    #Main change from tutorial - don't calculate the means of the embeddings extracted from each 500ms speech turn.
    
    #emb: chosen pretrained embedding extractor model
    #one_file: URI of .wav file in the format the model expects
    
    emb_from_sample = []
    
    # obtain raw embeddings (as `pyannote.core.SlidingWindowFeature` instance)
    embeddings = emb(one_file)

    # obtain raw SAD scores (as `pyannote.core.SlidingWindowFeature` instance), binarize raw SAD scores
    # NOTE: both onset/offset values were tuned on AMI dataset. Might need to use different values for better results
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

    #We only work of long (> ts) speech turns. Default is 2s from the tutorial
    t = 2
    long_turns = Timeline(segments=[s for s in speech_turns if s.duration > t])

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

def resample_data(X, factor, dropNA = True):
    if dropNA:
        X = X.dropna(inplace = False)
    
    #shuffle data within participant
    X = X.groupby('part').sample(frac=1).reset_index(drop=True)
    grp = X.groupby('part')
    Xdown = np.zeros(513)
    
    #mean every FACTOR datapoints per participant, drop the carryover
    for name, features in grp:
            p = features.part
            if 'sno' in features:
                features = features.drop('sno', axis=1)
            if 'part' in features:
                features = features.drop('part', axis=1)
            if 'cond' in features:
                features = features.drop('cond', axis=1)
            features = features.to_numpy(dtype = 'float32')
            downsamp = block_reduce(features, block_size=(factor, 1), func=np.mean, cval=np.nan)
            downsamp = np.where(np.isfinite(downsamp), downsamp, pd.NA)
            downsamp = np.concatenate((downsamp, np.full((downsamp.shape[0],1), name)), axis = 1)
            Xdown = np.vstack((Xdown, downsamp)) 
    Xdown = pd.DataFrame(Xdown[1:], columns = np.append(np.arange(512)+1, 'part')).dropna()
    
    return Xdown

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
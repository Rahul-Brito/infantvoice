import os
import numpy as np
import pandas as pd

from skimage.measure import block_reduce

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
            downsamp = block_reduce(embeddings.drop(columns='part_id').to_numpy(dtype='float32'), block_size=(factor, 1), func=np.mean, cval=np.nan)

            emb_down.append(downsamp)#append each participants resampled embeddings
            emb_down['part_id'] = embeddings.part_id.reset_index()
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

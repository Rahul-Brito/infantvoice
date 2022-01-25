import os
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

def pyannote_run_directory(directory, sid_creator, save_dir, save_name, save=False):

    #extracts embeddings from wav files in a folder
    #does speaker activity and change detection for long sentences, but just embedding extraction for concatenated hellos    
    all_embs, sid_per_sample, stim_type = [], [], []

    print("Started")
    for filename in os.listdir(directory):
        if filename.endswith(".wav"): 
            clear_output(wait=True)
            
            print("Processing" + str(filename))
            
            one_file = {'uri': 'filename', 'audio': os.path.join(directory, filename)}
            
            all_embs,sid_per_sample,stim_type,test_inter = pyannote_extract_embs(one_file, all_embs, sid_per_sample, sid_creator)

            

    all_embs = pd.DataFrame(np.vstack(all_embs))
    all_embs['part'] = sid_per_sample
    all_embs['stim_type'] = stim_type
    print("Done")
    
    if save:
        all_embs.to_csv(os.path.join(save_dir, save_name))

    return all_embs,sid_per_sample,stim_type,test_inter



def pyannote_extract_embs(one_file, all_embs, sid_per_sample, sid_creator):
    
    
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
        all_embs.append(inter)
        sid_per_sample,stim_type = sid_creator(inter,sid_per_sample, stim_type)
    
    return all_embs, id_per_sample, stim_type, test_inter
        

    
    
    
def readings_sid_creator(inter,sid_per_sample, stim_type):
    for i in range(0, inter.shape[0]):
        sid_per_sample.append(filename[0:2])
    return sid_per_sample,stim_type




def VFPpara_sid_creator(inter,sid_per_sample,stim_type):
    for i in range(0, inter.shape[0]):
        p = filename[filename.index('_')-2:filename.index('_')]
        vtype = filename[filename.index('_')+1]
        idn = os.path.splitext(filename)[0][-1]
        if idn=='h':
            idn=''
        stim_type.append(vtype+idn)
        if 'Norm' in filename:
            sid_per_sample.append(p + 'n')
        else:
            sid_per_sample.append(p + 'v')
    return sid_per_sample,stim_type
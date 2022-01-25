def pyannote_extract_embs(one_file, all_embs, sid_per_sample, sid_creator)
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
        sid_per_sample = sid_creator
    
    return all_embs, sid_per_sample
        
def readings_sid_creator(inter,sid_per_sample):
    for i in range(0, inter.shape[0]):
        sid_per_sample.append(filename[0:2])
    return sid_per_sample


def extract_embs_readings(directory, samptype, save_dir, save_name, save=False):

    #extracts embeddings from wav files in a folder
    #does speaker activity and change detection for long sentences, but just embedding extraction for concatenated hellos

    all_embs, sid_per_sample, sno = [], [], []

    print("Started")
    for filename in os.listdir(directory):
        if filename.endswith(".wav"): 
            clear_output(wait=True)
            one_file = {'uri': 'filename', 'audio': os.path.join(directory, filename)}

            all_embs,sid_per_sample = pyannote_extract_embs(one_file, all_embs, sid_per_sample, sid_creator)

            print("Processing" + str(filename))

    all_embs = pd.DataFrame(np.vstack(all_embs))
    all_embs['part'] = sid_per_sample
    print("Done")
    
    if save:
        all_embs.to_csv(os.path.join(save_dir, save_name))

    return all_embs




def extract_embeddings(directory, samptype, save_dir, save_name, save=False):

    #extracts embeddings from wav files in a folder
    #does speaker activity and change detection for long sentences, but just embedding extraction for concatenated hellos

    X, sid_per_sample, inter, sno = [], [], [], []

    print("Started")
    for filename in os.listdir(directory):
        if filename.endswith(".wav"): 
            clear_output(wait=True)
            one_file = {'uri': 'filename', 'audio': os.path.join(directory, filename)}

            # obtain raw embeddings (as `pyannote.core.SlidingWindowFeature` instance)
            embeddings = emb(one_file)

            if samptype == 'concat hellos':
                #X.append(np.mean(embeddings, axis=0))
                X.append(embeddings.data)
                for i in range(0, embeddings.data.shape[0]):
                    sid_per_sample.append(filename[0:2])

            elif 'paralysis' in samptype:
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
                    #X.append(np.mean(inter, axis=0))
                    X.append(inter)
                    for i in range(0, inter.shape[0]):
                        p = filename[filename.index('_')-2:filename.index('_')]
                        vtype = filename[filename.index('_')+1]
                        idn = os.path.splitext(filename)[0][-1]
                        if idn=='h':
                            idn=''
                        sno.append(vtype+idn)
                        if 'Norm' in filename:
                            sid_per_sample.append(p + 'n')
                        else:
                            sid_per_sample.append(p + 'v')
                        
                        
            
            elif samptype == 'long recording' or samptype == 'toy':            
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
                    #X.append(np.mean(inter, axis=0))
                    X.append(inter)
                    for i in range(0, inter.shape[0]):
                        sid_per_sample.append(filename[0:2])


            # keep track of speaker label (for later scatter plot)
            #sid_per_sample.append(filename[0:2])
            #sid.append(filename[0:4])
            print("Processing" + str(filename))

    X = pd.DataFrame(np.vstack(X))
    X['part'] = sid_per_sample
    if samptype == 'paralysis':
        X['sno'] = sno
    print("Done")
    
    if save:
        X.to_csv(os.path.join(save_dir, save_name))

    #outputs embeddings in a dataframe with participant labels per row
    return X




def extract_embeddings(directory, samptype, save_dir, save_name, save=False):

    #extracts embeddings from wav files in a folder
    #does speaker activity and change detection for long sentences, but just embedding extraction for concatenated hellos

    X, sid_per_sample, inter, sno = [], [], [], []

    print("Started")
    for filename in os.listdir(directory):
        if filename.endswith(".wav"): 
            clear_output(wait=True)
            one_file = {'uri': 'filename', 'audio': os.path.join(directory, filename)}

            # obtain raw embeddings (as `pyannote.core.SlidingWindowFeature` instance)
            embeddings = emb(one_file)

            if samptype == 'concat hellos':
                #X.append(np.mean(embeddings, axis=0))
                X.append(embeddings.data)
                for i in range(0, embeddings.data.shape[0]):
                    sid_per_sample.append(filename[0:2])

            elif 'paralysis' in samptype:
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
                    #X.append(np.mean(inter, axis=0))
                    X.append(inter)
                    for i in range(0, inter.shape[0]):
                        p = filename[filename.index('_')-2:filename.index('_')]
                        vtype = filename[filename.index('_')+1]
                        idn = os.path.splitext(filename)[0][-1]
                        if idn=='h':
                            idn=''
                        sno.append(vtype+idn)
                        if 'Norm' in filename:
                            sid_per_sample.append(p + 'n')
                        else:
                            sid_per_sample.append(p + 'v')
                        
                        
            
            elif samptype == 'long recording' or samptype == 'toy':            
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
                    #X.append(np.mean(inter, axis=0))
                    X.append(inter)
                    for i in range(0, inter.shape[0]):
                        sid_per_sample.append(filename[0:2])


            # keep track of speaker label (for later scatter plot)
            #sid_per_sample.append(filename[0:2])
            #sid.append(filename[0:4])
            print("Processing" + str(filename))

    X = pd.DataFrame(np.vstack(X))
    X['part'] = sid_per_sample
    if samptype == 'paralysis':
        X['sno'] = sno
    print("Done")
    
    if save:
        X.to_csv(os.path.join(save_dir, save_name))

    #outputs embeddings in a dataframe with participant labels per row
    return X
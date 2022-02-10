import numpy as np
import pandas as pd

from IPython.display import clear_output

from scipy.spatial.distance import directed_hausdorff, euclidean, cosine

def hausdorff_distances(emb_2d):
    # directed or assymetric variant     
    labels = emb_2d.part_id.unique()
    len_labels = len(labels)
    # build empty df
    pairwise_distances_hausdorff = pd.DataFrame(np.zeros((len_labels, len_labels)) , columns = labels, index=labels)
    # pairwise_distances_euclidean = pd.DataFrame(np.zeros((len_labels, len_labels)) , columns = labels, index=labels)                          

    # Build df out of X
    #df = pd.DataFrame(X)
    #df.columns = ['x1', 'x2']
    #df['label'] = y

    # Compute pairwise distance between labelled arrays 
    for row in range(len_labels):
        for col in range(len_labels):
            clear_output(wait=True)
            label_a = labels[row]
            label_b = labels[col]
            label_a_values = emb_2d[emb_2d.part_id==label_a][['dim0','dim1']].values
            label_b_values = emb_2d[emb_2d.part_id==label_b][['dim0','dim1']].values
            dist_hausdorff = directed_hausdorff(label_a_values,label_b_values)
            pairwise_distances_hausdorff.iloc[row,col]= dist_hausdorff[0]
            print("Processing row " + str(row) + ", col " + str(col))
    #         dist_euclidean = euclidean(label_a_values.mean(axis=0),label_b_values.mean(axis=0))
    #         pairwise_distances_euclidean.iloc[row,col]= dist_euclidean

    #normalizes cos distances to max distance
    max_haus = pairwise_distances_hausdorff.to_numpy().max()
    pairwise_distances_hausdorff = pairwise_distances_hausdorff.divide(max_haus)
    return pairwise_distances_hausdorff



def euclidean_distances(X,y, labels):
    # directed or assymetric variant     
    len_labels = len(labels)
    # build empty df
    pairwise_distances_euclidean = pd.DataFrame(np.zeros((len_labels, len_labels)) , columns = labels, index=labels)                          

    # Build df out of X
    df = pd.DataFrame(X)
    df.columns = ['x1', 'x2']
    df['label'] = y

    # Compute pairwise distance between labelled arrays 
    for row in range(len_labels):
        for col in range(len_labels):
            clear_output(wait=True)
            label_a = labels[row]
            label_b = labels[col]
            label_a_values = df[df.label==label_a][['x1','x2']].values
            label_b_values = df[df.label==label_b][['x1','x2']].values
            dist_euclidean = euclidean(label_a_values.mean(axis=0),label_b_values.mean(axis=0))
            pairwise_distances_euclidean.iloc[row,col]= dist_euclidean


    return pairwise_distances_euclidean


def cos_distance(emb_a):
    # directed or assymetric variant     
    labels = emb_a.part_id.unique()
    len_labels = len(labels)
    # build empty df
    pairwise_distances_cosine = pd.DataFrame(np.zeros((len_labels, len_labels)) , columns = labels, index=labels)
    # pairwise_distances_euclidean = pd.DataFrame(np.zeros((len_labels, len_labels)) , columns = labels, index=labels)                          

    # Compute pairwise distance between labelled arrays 
    for row in range(len_labels):
        for col in range(len_labels):
            clear_output(wait=True)
            label_a = labels[row]
            label_b = labels[col]
            label_a_values = emb_a[emb_a.part_id==label_a].values
            label_b_values = emb_a[emb_a.part_id==label_b].values
            dist_cos = cosine(label_a_values,label_b_values)
            pairwise_distances_cosine.iloc[row,col]= dist_cos
            print("Processing row " + str(row) + ", col " + str(col))
    
    #normalize to largest
    max_cos = pairwise_distances_cosine.to_numpy().max()
    pairwise_distances_cosine = pairwise_distances_cosine.divide(max_cos)
    
    return pairwise_distances_cosine
        
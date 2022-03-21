import numpy as np
import pandas as pd

from IPython.display import clear_output

import scipy
from scipy.spatial.distance import directed_hausdorff, euclidean, cosine, pdist
from scipy import stats

from sklearn.neighbors import NearestNeighbors


def hausdorff_distances(emb_2d):
    # directed or assymetric variant     
    labels = emb_2d.part_id.unique().astype('int32')
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


def embedding_quality(X, Z, classes, knn=10, knn_classes=10, subsetsize=1000):
    #taken directly from https://github.com/berenslab/rna-seq-tsne/blob/master/toy-example.ipynb
    
    nbrs1 = NearestNeighbors(n_neighbors=knn).fit(X)
    ind1 = nbrs1.kneighbors(return_distance=False)

    nbrs2 = NearestNeighbors(n_neighbors=knn).fit(Z)
    ind2 = nbrs2.kneighbors(return_distance=False)

    intersections = 0.0
    for i in range(X.shape[0]):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn = intersections / X.shape[0] / knn
    
    cl, cl_inv = np.unique(classes, return_inverse=True)
    C = cl.size
    mu1 = np.zeros((C, X.shape[1]))
    mu2 = np.zeros((C, Z.shape[1]))
    for c in range(C):
        mu1[c,:] = np.mean(X[cl_inv==c,:], axis=0)
        mu2[c,:] = np.mean(Z[cl_inv==c,:], axis=0)
        
    nbrs1 = NearestNeighbors(n_neighbors=knn_classes).fit(mu1)
    ind1 = nbrs1.kneighbors(return_distance=False)
    nbrs2 = NearestNeighbors(n_neighbors=knn_classes).fit(mu2)
    ind2 = nbrs2.kneighbors(return_distance=False)
    
    intersections = 0.0
    for i in range(C):
        intersections += len(set(ind1[i]) & set(ind2[i]))
    mnn_global = intersections / C / knn_classes
    
    subset = np.random.choice(X.shape[0], size=subsetsize, replace=False)
    d1 = pdist(X[subset,:])
    d2 = pdist(Z[subset,:])
    rho = scipy.stats.spearmanr(d1[:,None],d2[:,None]).correlation
    
    return (mnn, mnn_global, rho)

def loo_t_test(loo_dict,equal_var):
    #pass in a dictionary <loo_dict> of format {participant-removed:flattened distance}
    #calculate t-test between all this distribution to test for the null hypothesis that 2 independent distributions of distances have identical average (expected) values.
    
    #initialize blank array
    len_labels=len(loo_dict)
    labels=list(loo_dict.keys())
    pairwise_ttest = pd.DataFrame(np.zeros((len_labels, len_labels)), columns=labels, index=labels)
    
    for row in range(len_labels):
        for col in range(len_labels):
            clear_output(wait=True)
            label_a = labels[row]
            label_b = labels[col]
            label_a_values = loo_dict[label_a]
            label_b_values = loo_dict[label_b]
            ttest = stats.ttest_ind(label_a_values,label_b_values,equal_var=equal_var)
            pairwise_ttest.iloc[row,col]= ttest[1]
            print("Processing row " + str(row) + ", col " + str(col))
            
    return pairwise_ttest
        
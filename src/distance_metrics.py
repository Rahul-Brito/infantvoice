import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff, euclidean, cosine, pdist
from IPython.display import clear_output

import scipy

from scipy import stats

from sklearn.neighbors import NearestNeighbors


def hausdorff_distances(emb_2d):
    # directed or assymetric variant     
    labels = emb_2d.part_id.unique()
    len_labels = len(labels)
    
    # build empty df
    pairwise_distances_hausdorff = pd.DataFrame(np.zeros((len_labels, len_labels)) , columns = labels, index=labels)

    # Compute pairwise distance between labelled arrays 
    for row in range(len_labels):
        for col in range(len_labels):
            clear_output(wait=True)
            label_a = labels[row]
            label_b = labels[col]

            label_a_values = emb_2d[emb_2d.part_id==label_a].drop(columns='part_id').to_numpy()
            label_b_values = emb_2d[emb_2d.part_id==label_b].drop(columns='part_id').to_numpy()
            
            dist_hausdorff = directed_hausdorff(label_a_values,label_b_values)

            if row != col:
                pairwise_distances_hausdorff.iloc[row,col]= dist_hausdorff[0]
            else:
                pairwise_distances_hausdorff.iloc[row,col]= np.nan
            print("Processing row " + str(row) + ", col " + str(col))

    pairwise_distances_hausdorff_zscore = pd.DataFrame(
        stats.zscore(pairwise_distances_hausdorff.to_numpy(), 
                     axis=None, ddof=0, nan_policy='omit'),
        columns = labels, index=labels)

    return pairwise_distances_hausdorff_zscore


def cos_distance(emb_a):
    # directed or assymetric variant     
    labels = emb_a.part_id.unique().astype('int32')
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
            label_a_values = emb_a[emb_a.part_id==label_a].drop(columns='part_id').to_numpy()
            label_b_values = emb_a[emb_a.part_id==label_b].drop(columns='part_id').to_numpy()
            dist_cos = cosine(label_a_values,label_b_values)
            if label_a != label_b:
                pairwise_distances_cosine.iloc[row,col]= dist_cos
            else:
                pairwise_distances_cosine.iloc[row,col]= np.nan
               
            print("Processing row " + str(row) + ", col " + str(col))
    
    #z-score
    pairwise_distances_cosine = pairwise_distances_cosine.apply(stats.zscore, nan_policy='omit')
    
    return pairwise_distances_cosine







def euclidean_distances(X):
    # directed or assymetric variant     
    labels = X.part_id.unique().astype('int32')
    len_labels = len(labels)
    # build empty df
    pairwise_distances_euclidean = pd.DataFrame(np.zeros((len_labels, len_labels)) , columns = labels, index=labels)                          

    # Build df out of X
    #df = pd.DataFrame(X)
    #df.columns = ['x1', 'x2']
    #df['label'] = y

    df = X
    # Compute pairwise distance between labelled arrays 
    for row in range(len_labels):
        for col in range(len_labels):
            clear_output(wait=True)
            label_a = labels[row]
            label_b = labels[col]
            label_a_values = df[df.part_id==label_a].drop(columns='part_id').to_numpy()
            label_b_values = df[df.part_id==label_b].drop(columns='part_id').to_numpy()
            dist_euclidean = euclidean(label_a_values.mean(axis=0),label_b_values.mean(axis=0))
            if label_a != label_b:
                pairwise_distances_euclidean.iloc[row,col]= dist_euclidean
            else:
                pairwise_distances_euclidean.iloc[row,col]= np.nan


    return pairwise_distances_euclidean




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
        
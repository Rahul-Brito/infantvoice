import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import umap

def run_tsne(emb_down, perplexity = 30, metric="euclidean"):

    tsne = TSNE(n_components=2, perplexity=perplexity, metric=metric, init = 'pca', square_distances=True)

    emb_tsne = pd.DataFrame(tsne.fit_transform(emb_down.drop(columns='part_id')), columns = ['dim0', 'dim1'])
    emb_tsne['part_id'] = emb_down['part_id']
    dimreduc = 'tSNE'

    return emb_tsne



def run_umap(X=None, y=None, method = 'unsupervised', scaler=None, neighbor = 10, dist=0.1, metric='correlation', 
             color_code = None, annotate_names = None, annotate = False, test_set = True, title=None, 
             savefig_path = False, X_test=None, y_test=None, color_code_test = None, plot=True, plot_centroid_names = True):
    
    dimreduc = 'UMAP'
    #reducer = umap.UMAP(n_components=dimension, n_neighbors = neighbor, min_dist=dist,metric=metric,random_state=seed_value) #, TSNE(n_components=k, random_state=seed_value), PCA(n_components=k, random_state=seed_value)]
    reducer = umap.UMAP()
    reducer_name = 'umap' #, 'tsne', 'pca']
    X_reduced = reducer.fit_transform(X)

   # #pipeline = Pipeline([
    #    ('normalization', scaler),
    #    ('reducer', reducer), ])

    #y_encoded = LabelEncoder().fit_transform(y)
    #if method == 'supervised':
    #    X_reduced = reducer.fit_transform(X, y_encoded)
    #elif method == 'metric_learning':
        #train
        #mapper = reducer.fit(X, y_encoded)
    #    X_reduced = reducer.fit_transform(X, y_encoded)
    #    #test
    #    X_reduced_test = reducer.transform(X_test)
        
    #elif method == 'unsupervised':
        #X_reduced = reducer.fit_transform(X)

    # find centroids and labels
    data = pd.DataFrame(X_reduced, columns = ['dim0', 'dim1'])
    data['part_id'] = X['part_id']
    #data['label'] = y

    #centers = []
    #centers_labels = np.unique(y)
    #for sr in centers_labels:
    #    data_sr = data[data.label==sr]
    #    sr_centroid = data_sr[['dim0','dim1']].mean(axis=0)
    #    centers.append(sr_centroid)
    
    #centers = np.array(centers)
    
    #data.drop('label', axis=1, inplace = True)
    # Plot in 2D
    #if plot:
    #    assert dimension == 2 
    #    if method == 'metric_learning':
            # train: first time point
            #scatter_plot(X_reduced, y, color_code, method, annotate = annotate, title = 'First time step (train set)', savefig_path = savefig_path )
    #        scatter_plot(X_reduced, y, color_code, method, annotate = annotate, title = 'First time step (train set)')
            # test: next time points            
            #scatter_plot(X_reduced_test, y_test, color_code_test, method, annotate = annotate, title = title, savefig_path = savefig_path)
    #        scatter_plot(X_reduced_test, y_test, color_code_test, method, annotate = annotate, title = title)
            
    #    else:
    #        scatter_plot(X_reduced, y, color_code, method, annotate = annotate, title = title, savefig_path = savefig_path, centers=centers, centers_labels = centers_labels,plot_centroid_names = plot_centroid_names )
    if method == 'metric_learning':
        return data, pd.DataFrame(X_reduced_test, columns = ['dim0', 'dim1']), centers, centers_labels
    else:
        return data#, centers, centers_labels#, dimreduc
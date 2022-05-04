import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import umap
import pacmap

def run_tsne(emb_down,random_state=42, perplexity = 30,init='random',learning_rate=200, metric="euclidean"):

    #run tsne on the data and return a pd with data label. DEFAULT IS ONE RANDOM SEED instead of different random generations
    tsne = TSNE(n_components=2, perplexity=perplexity, metric=metric, init = init, learning_rate=learning_rate, square_distances=True)

    emb_tsne = pd.DataFrame(tsne.fit_transform(emb_down.drop(columns='part_id')), columns = ['dim0', 'dim1'])
    emb_tsne['part_id'] = emb_down['part_id'].to_numpy()
    dimreduc = 'tSNE'

    return emb_tsne



def run_umap(emb_down, n_components=2, n_neighbors = 10, dist=0.1, metric='correlation'):
    
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors)
    X_reduced = reducer.fit_transform(emb_down)

    col_lab = ['dim' + str(d) for d in np.arange(n_components)]
    data = pd.DataFrame(X_reduced, columns = col_lab)
    data['part_id'] = emb_down['part_id'].to_numpy()
    
    return data

def run_pacmap(emb_down, n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0):
    
    # initializing the pacmap instance
    # Setting n_neighbors to "None" leads to a default choice shown below in "parameter" section
    embedder_pacmap = pacmap.PaCMAP(n_dims, n_neighbors, MN_ratio, FP_ratio) 

    # fit the data (The index of transformed data corresponds to the index of the original data)
    emb_pacmap = embedder_pacmap.fit_transform(emb_down.drop(columns='part_id').to_numpy(), init="pca")

    data = pd.DataFrame(emb_pacmap, columns = ['dim0', 'dim1'])
    data['part_id'] = emb_down['part_id'].to_numpy()
    
    return data
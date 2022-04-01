import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import umap

def run_tsne(emb_down,random_state=42, perplexity = 30,init='random',learning_rate=200, metric="euclidean"):

    #run tsne on the data and return a pd with data label. DEFAULT IS ONE RANDOM SEED instead of different random generations
    tsne = TSNE(n_components=2, perplexity=perplexity, metric=metric, init = init, learning_rate=learning_rate, square_distances=True)

    emb_tsne = pd.DataFrame(tsne.fit_transform(emb_down.drop(columns='part_id')), columns = ['dim0', 'dim1'])
    emb_tsne['part_id'] = emb_down['part_id'].to_numpy()
    dimreduc = 'tSNE'

    return emb_tsne



def run_umap(emb_down, neighbor = 10, dist=0.1, metric='correlation'):
    
    reducer = umap.UMAP()
    X_reduced = reducer.fit_transform(emb_down)

    data = pd.DataFrame(X_reduced, columns = ['dim0', 'dim1'])
    data['part_id'] = emb_down['part_id'].to_numpy()
    
    return data
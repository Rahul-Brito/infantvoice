def plot_2Ddata(X2d, dimreduc, samptype, colors):
    fig, ax = plt.subplots(1, figsize=(11,9))
    #for x, y, w, t in zip(X2d.dim0, X2d.dim1, X2d.first_sample, X2d.index):
    
    if 'paralysis' in samptype:
        for x, y, t in zip(X2d.dim0, X2d.dim1, X2d['part'].str.slice(0,2).to_numpy(dtype = 'int32')):
        #    #plt.text(x, y, w, color=colors[t-1], fontsize=10)
            plt.text(x, y, str(t), color=colors[t-1], fontsize=10)
        plt.scatter(X2d.dim0, X2d.dim1, c='1', alpha = 0.6, s=10)
        #sns.scatterplot(X2d.dim0, X2d.dim1, hue=X2d['part'].str.slice(0,2), palette=colors, alpha = 0.6, s=10)

    else:
        for x, y, t in zip(X2d.dim0, X2d.dim1, X2d.part):
            #plt.text(x, y, w, color=colors[t-1], fontsize=10)
            plt.text(x, y, str(t), color=colors[t-1], fontsize=10)
        plt.scatter(X2d.dim0, X2d.dim1, c='1', alpha = 0.6, s=10)
    if dimreduc == 'tSNE':
        plt.title("Plot of participant using tSNE, perplexity=" + str(per) + ", with metric=" + met + ", for " + samptype)
    elif dimreduc == 'UMAP':
        plt.title("Plot of participant using UMAP for " + samptype)
    plt.show()

def gen_color_code(X2d, name):
    colors = sns.color_palette("Paired", len(X2d['part'].unique())+1)
    #colors[13:]=sns.color_palette("hls")
    #colors[10] = (0,0,0)
    cmap = {}
    #[cmap.update({z:colors[z]}) for z in np.array(X2d.part.unique())]
    [cmap.update({z:colors[z]}) for z in np.arange(X2d['part'].unique().shape[0])]
    #.str.slice(0,2).to_numpy(dtype = 'int32')
    #X2d['c_clust'] = X2d.clusterSC.map(cmap)
    
    if 'paralysis' in name:
        X2d['c_part'] = [colors[z-1] for z in X2d['part'].str.slice(0,2).to_numpy(dtype = 'int32')]
    else:
        X2d['c_part'] = [colors[z-1] for z in X2d['part'].str.slice(0,2)]
    
    return X2d, colors 


def get_cluster_data(x, datasetname, factor,  umap_metric, resample=True, plot=True):
    #hellos2D_tsne = run_tSNE(hdown.drop('part', axis=1), perplexity = 30, metric = "euclidean")
    #sent2D_tsne = run_tSNE(sdown.drop('part', axis=1), perplexity = (len(sentdown.index)//10), metric="euclidean")
      
    if resample:
        xdown = resample_data(x, factor)
    else:
        xdown = x
   
    #run UMAP on xdown
    x2d, centers, center_labels = run_umap(X=xdown.drop('part',axis=1), y=xdown['part'], method = 'unsupervised', plot=False, metric=umap_metric)
    dimreduc = 'UMAP'
    #x2d['part'] = xdown['part'].str.slice(0,2).to_numpy(dtype = 'int32')
    x2d = x2d.dropna(inplace=False)
    x2d['part'] = xdown['part'].to_numpy()
    x2d, colors = gen_color_code(x2d, datasetname)
    
    if plot:
        plot_2Ddata(x2d, dimreduc, datasetname, colors)
    
    return {datasetname:{'x2d': x2d, 'centers':centers, 'center_labels':center_labels, 'colors': colors, 'data_resamp': xdown}}
    #return {datasetname:{'data_2D': x2d, 'centers':centers, 'center_labels':center_labels, 'data_resamp': xdown}}

    
def plot_norm_heatmap(dist_matrix, samptype):    
    #takes in normalized data and plots it
    
    # Generate a mask for the upper triangle for each
    mask = np.triu(np.ones_like(dist_matrix, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    ax.xaxis.set_ticks_position("top")

    # Generate a custom diverging colormap
    hmapcol = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(dist_matrix, mask=mask, vmin=-1.0, vmax=1.0, cmap=hmapcol, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Distances between centroids of participants, for" + samptype, y = 1.08,fontsize=12)

#fig, ax = plt.subplots(1, figsize=(10,8))
#for x, y, t, g in zip(X2d.dim0, X2d.dim1, X2d.index, X2d.gcol):
#    plt.text(x, y, str(t), color=g, fontsize=10)
#plt.scatter(X2d.dim0, X2d.dim1, c='1', alpha = 0.6, s=10)
#plt.title("Plot of participant by perceived gender" + str(seedno))
#plt.show()



import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import sklearn.decomposition as skd
from processinglibs import processing




if __name__ == '__main__':
    ###### generate simulated data ######

    ###### cluster profiles directly ######
    profiles, classes, diffs = processing.get_data(return_classes=True)
    profile_mult = classes[0]
    diffs = diffs[0]
    # rows are colored by whether they were in one class (i.e. diffs*1) or the other class
    # (diffs*-1)
    colors = sns.color_palette('Paired', n_colors=2)
    row_colors = [colors[0] if factor==-1 else colors[1] for factor in profile_mult]
    
    # hierarchically cluster
    z = sch.linkage(profiles, method='weighted')
    
    # plot the heatmap with the dendrogram.
    cg = sns.clustermap(profiles, row_linkage=z, col_cluster=False, cmap='coolwarm',
                        row_colors=row_colors, vmin=-2, vmax=2)
    
    # change rotation of xtick and ytick labels
    plt.setp(cg.ax_heatmap.get_xmajorticklabels(), rotation=90)
    plt.setp(cg.ax_heatmap.get_ymajorticklabels(), rotation=0)
    plt.subplots_adjust(right=0.8) # makes the yticks visible
    plt.savefig('clustermap.pdf')
    
    
    ###### cluster the  principal components ######
    pca, transformed, loadings = processing.doPCA(profiles)
    
    # plot explained variance by each PC.
    plt.figure(figsize=(4,4));
    pd.Series(pca.explained_variance_ratio_).plot(kind='bar');
    plt.ylabel("fraction of variance \nexplained by each PC", fontsize=14);
    plt.tight_layout()
    plt.savefig('barplot.variance_explained_by_PCs.pdf')
    
    # plot the correlation between 'diffs' and principal components
    correlation = pd.Series({idx:np.abs(st.pearsonr(diffs, loading_vec)[0]) for idx, loading_vec in loadings.iterrows()})
    plt.figure(figsize=(4,4));
    correlation.plot(kind='bar');
    plt.ylabel("abs. value of correlation between\n'diffs' and each PC", fontsize=14);
    plt.tight_layout()
    plt.savefig('barplot.correlation_with_designed_diffs.pdf')

    # now cluster the profiles by their top principal component
    z_pca = sch.linkage(transformed.loc[:, ['pc_0']], method='weighted')
    cg = sns.clustermap(profiles, row_linkage=z_pca, col_cluster=False,
                        cmap='coolwarm',
                        row_colors=row_colors,  vmin=-2, vmax=2)
    plt.setp(cg.ax_heatmap.get_xmajorticklabels(), rotation=90)
    plt.setp(cg.ax_heatmap.get_ymajorticklabels(), rotation=0)
    plt.subplots_adjust(right=0.8) # makes the yticks visible
    plt.savefig('clustermap_by_PC1.pdf')

    # plot diffs and PCs
    diffs_norm = diffs/np.sqrt((diffs**2).sum())
    pd.concat([diffs_norm.rename('designed_diff'), loadings.iloc[0].rename('PC1')], axis=1).plot(kind='bar')
    plt.savefig('barplot.designed_diff_and_PC1_values.pdf')


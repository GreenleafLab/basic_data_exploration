import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import sklearn.decomposition as skd

###### functions ######
def doPCA(data):
    """Perform PCA on a matrix."""
    PCA = skd.PCA(n_components=None, whiten=False)
    transformed_data = PCA.fit_transform(data)
    principal_comp = np.dot(np.linalg.inv(np.dot(transformed_data.T, transformed_data)),
                                 np.dot(transformed_data.T, data))
    
    # put in terms of pandas dataframe
    transformed_data = pd.DataFrame(transformed_data, index=data.index,
                                    columns=['pc_%d'%i for i in np.arange(transformed_data.shape[1])])
    principal_comp = pd.DataFrame(principal_comp, columns=data.columns,
                                  index=['pc_%d'%i for i in np.arange(transformed_data.shape[1])])
    
    return (PCA, transformed_data, principal_comp)

###### generate simulated data ######
# define an average profile, i.e. in dG space from gaussian distributed noise
avg_profile = pd.Series(np.hstack([st.norm.rvs(loc=-10, size=3),
                                   st.norm.rvs(loc=-8, size=5)]),
                        index=['context_%d'%i for i in range(8)])

# define the average difference from the average profile of a "class" (other class
# has -1 times this difference)
diffs = pd.Series([-1, 0.5, 0.5, -0.25, -0.25, 0.5, 0.5, 0], index=avg_profile.index)

# randomly choose the classes for 15 observations from 1 or -1
profile_mult = pd.Series([-1]*6 + [1]*9,
                         index=['observation_%02d'%i for i in range(15)])

# add error to each observation
error = pd.DataFrame(st.norm.rvs(loc=0, scale=0.25, size=[15, len(diffs)]),
                     columns=avg_profile.index, index=profile_mult.index)

# generate profiles by adding error and difference (times 1 or -1) to the avg profile
profiles = pd.concat({idx:avg_profile + factor*diffs + err_vec
                      for factor, (idx, err_vec) in zip(profile_mult, error.iterrows())}).unstack()

###### cluster profiles directly ######

# rows are colored by whether they were in one class (i.e. diffs*1) or the other class
# (diffs*-1)
colors = sns.color_palette('Paired', n_colors=2)
row_colors = [colors[0] if factor==-1 else colors[1] for factor in profile_mult]

# hierarchically cluster
z = sch.linkage(profiles, method='weighted')

# plot the heatmap with the dendrogram.
cg = sns.clustermap(profiles-avg_profile, row_linkage=z, col_cluster=False, cmap='coolwarm',
                    row_colors=row_colors, vmin=-2, vmax=2)

# change rotation of xtick and ytick labels
plt.setp(cg.ax_heatmap.get_xmajorticklabels(), rotation=90)
plt.setp(cg.ax_heatmap.get_ymajorticklabels(), rotation=0)
plt.subplots_adjust(right=0.8) # makes the yticks visible
plt.savefig('clustermap.pdf')


###### cluster the  principal components ######
pca, transformed, loadings = doPCA(profiles)

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
cg = sns.clustermap(profiles-avg_profile, row_linkage=z_pca, col_cluster=False,
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


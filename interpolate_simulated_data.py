import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import sklearn.decomposition as skd
from processinglibs import processing
from tectolibs import clustering



if __name__ == '__main__':
    ###### generate simulated data ######
    data, classes, loadings = processing.get_data(num_observations=100,
                                                  context_diff_scale=[2, 1],
                                                  num_pcs=2,
                                                  error_scale=0.2,
                                                  return_classes=True)
    
    # make some data points go missing
    num_missing = 20
    data_stacked = data.stack().copy()
    index_missing = data_stacked.iloc[np.random.choice(np.arange(len(data_stacked)), size=num_missing, replace=False)].sort_index().index.tolist()
    data_stacked.loc[index_missing] = np.nan
    data_missing = data_stacked.unstack()
    
    # interpolate
    data_interp, interp_info = clustering.interpolate_mat_knn(data_missing)
    
    # compare interpolated and original values
    interp_values = data_interp.stack().loc[index_missing]
    original_values = data.stack().loc[index_missing]
    interp_info_sub = pd.concat({s:interp_info.loc[s[0]] for s in index_missing}).unstack()
    
    # plot
    data_to_plot = pd.concat([original_values.rename('original_val'), interp_values.rename('interp_val'), interp_info_sub], axis=1)
    g = sns.FacetGrid(data_to_plot, hue='num_neighbors', palette='viridis'); g.map(plt.scatter, 'original_val', 'interp_val')
    plt.plot([-2, 2], [-2, 2], 'k--')


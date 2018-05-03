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


def get_data(num_observations=15, num_contexts=8, num_pcs=1, context_diff_scale=[2], error_scale=0.25, return_classes=False):
    """Return the simulated data.
    Inputs:
    num_observations: (N) number of different observations (i.e. rows).
    num_contexts: (M) number of different contexts associated with each observaion (i.e. columns).
    num_pcs: number of "principal components" differentiating the observations.
    context_diff_scale: list giving the scale (std dev) of the differences from zero for each pc.
    error_scale: scale (std. dev) of error.
    
    Returns:
    data: MxN matrix giving simulated data.
    """
    # check input
    if len(context_diff_scale)!=num_pcs:
        raise KeyError("must have same number of entries in 'context_diff_scale' as there are num_pcs")
    
    # set the names of the output data matrix
    index = ['observation_%02d'%i for i in range(num_observations)]
    names = ['context_%d'%i for i in range(num_contexts)]

    # define an average profile, i.e. in dG space from gaussian distributed noise
    init_mat = pd.DataFrame(0, columns=names, index=index)

    # add error to each observation
    error = pd.DataFrame(st.norm.rvs(loc=0, scale=error_scale, size=[num_observations, num_contexts]),
                         columns=names, index=index)
    
    # for each PC, find overall diff from average.
    classes = {}
    diffs = {}
    for i in range(num_pcs):
        # define a ddG profile associated with this pc.
        diff_profile = pd.Series(st.uniform.rvs(scale=context_diff_scale[i], size=num_contexts)-context_diff_scale[i]*0.5, index=names)
        diffs[i] = diff_profile
        
        # assign each observation to a class. Classes have equal numbers (num_observations/2 in each)
        whichclass = pd.Series(np.random.permutation([-1]*(num_observations/2) + [1]*(num_observations-num_observations/2)), index=index)
        classes[i] = whichclass # for every observation, defines which class it is in
        
        # matrix of deviations is the diff_profile*1 or -1 depending on the class
        deviations = pd.concat({idx:val*diff_profile for idx, val in whichclass.iteritems()}).unstack()
        init_mat = init_mat + deviations
    
    # final mat is deviations + error
    final_mat = init_mat + error
    
    if return_classes:
        return final_mat, pd.concat(classes, axis=1), pd.concat(diffs, axis=1)
    else:
        return init_mat + error
        
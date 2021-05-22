# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:43:20 2021

@author: A694772
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:52:44 2021

@author: Antoine
"""

# TODO : WIP

import numpy as np
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from CST.base_transformers.minirocket import MiniRocket
from CST.utils.checks_utils import check_array_3D
from CST.utils.shapelets_utils import generate_strides_2D, shapelet_dist_numpy, generate_strides_1D
from numba import set_num_threads, njit, prange
from numpy.lib.stride_tricks import as_strided

class ConvolutionalShapeletTransformer_tree_nb(BaseEstimator, TransformerMixin):
    def __init__(self, P=80, n_trees=200, max_ft=1.0, id_ft=0, use_class_weights=True,
                 verbose=0, n_bins=9, n_threads=3, random_state=None):
        """
        Initialize the Convolutional Shapelet Transform (CST)

        Parameters
        ----------
        P : array of int, optional
            Percentile used in the shapelet extraction process.
            The default is 80.
        n_trees : int or float, optional
        
        use_class_weights : bool, optional
        
        id_ft : int, optional
            Identifier of the feature on which the transform will be performed.
            The default is 0.
        verbose : int, optional
            Verbose parameter, higher values will output more logs. The default is 0.
        n_bins : int, optional
            Number of bins used in the candidates discretization. The default is 9.
        n_threads : int, optional
            Number of numba thread used. The default is 3.
        use_kernel_grouping : bool, optional
            Wheter or not to enable kernel grouping based on dilation and bias parameter.
            The default is True.
        random_state : int, optional
            Random state setter. The default is None.

        Returns
        -------
        None.

        """
        self.id_ft = id_ft
        self.verbose = verbose
        self.shapelets_params = None
        self.shapelets_values = None
        self.P = P
        self.n_trees = n_trees
        self.use_class_weights=use_class_weights
        self.max_ft = max_ft
        self.n_bins = n_bins
        self.n_threads = n_threads
        self.random_state = random_state

    def _log(self, message):
        if self.verbose > 0:
            print(message)

    def _log2(self, message):
        if self.verbose > 1:
            print(message)

    def fit(self, X, y, use_class_weights=True):
        """

        Parameters
        ----------
        X : array, shape = (n_samples, n_features, n_timestamps)
            Input data containing time series (tested with dtype np.float32), the algorithm
            will only process feature indicated by attribute id_ft.
        y : array, shape = (n_samples)
            Associated classes of the input time series
        use_class_weights : TYPE, optional
            Whether or not to balance computation based on the number of samples
            of each class.

        Returns
        -------
        ConvolutionalShapeletTransformer
            Fitted instance of self.

        """
        X = check_array_3D(X)
        set_num_threads(self.n_threads)
        #L = (n_samples, n_kernels, n_timestamps)
        # Kernel selection is performed in this function
        L, dils, biases, tree_splits = self._generate_inputs(X, y)
        X_indexes = np.zeros((len(tree_splits),X.shape[0]),dtype=np.int64)
        Y_indexes = np.zeros((len(tree_splits),X.shape[0]),dtype=np.int64)
        K_indexes = np.zeros((len(tree_splits)),dtype=np.int64)
        for i_split in range(len(tree_splits)):
            x_index, y_split, k_id = tree_splits[i_split]
            X_indexes[i_split,0:x_index.shape[0]] += x_index
            X_indexes[i_split,x_index.shape[0]:] -= 1
            Y_indexes[i_split,0:x_index.shape[0]] += y_split
            Y_indexes[i_split,x_index.shape[0]:] -= 1
            K_indexes[i_split] += k_id
        dils = dils[K_indexes]
        u_dils = np.unique(dils)
        shapelets = {}
        n_shapelets = 0
        for dil in u_dils:
            i_dil = np.where(dils==dil)[0]
            candidates_grp = process_all_nodes(X, y, L[:,i_dil], dil, self.P, 
                                               X_indexes[i_dil], Y_indexes[i_dil],
                                               K_indexes[i_dil])
            if candidates_grp.shape[0] > 0:
                candidates_grp = (candidates_grp - candidates_grp.mean(axis=-1, keepdims=True)) / (
                    candidates_grp.std(axis=-1, keepdims=True) + 1e-8)
                
                if not np.all(candidates_grp.reshape(-1, 1) == candidates_grp.reshape(-1, 1)[0]):
                    kbd = KBinsDiscretizer(n_bins=self.n_bins, strategy='uniform', dtype=np.float32).fit(
                        candidates_grp.reshape(-1, 1))
                    candidates_grp = np.unique(kbd.inverse_transform(
                        kbd.transform(candidates_grp.reshape(-1, 1))).reshape(-1, 9), axis=0)
                else:
                    candidates_grp = np.unique(candidates_grp, axis=0)
                shapelets.update({dil: candidates_grp})
                n_shapelets+=candidates_grp.shape[0]
        self.n_shapelets = n_shapelets
        self.shapelets_values = shapelets
        return self

    def transform(self, X):
        """
        Transform input time series into Shapelet distances

        Parameters
        ----------
        X : array, shape = (n_samples, n_features, n_timestamps)
            Input data containing time series (tested with dtype np.float32), the algorithm
            will only process feature indicated by attribute id_ft.

        Returns
        -------
        distances : array, shape = (n_samples, n_shapelets)
            Shapelet distance to all samples

        """
        self._check_is_fitted()
        X = check_array_3D(X)
        distances = np.zeros((X.shape[0], self.n_shapelets))
        prev = 0
        for i, dil in enumerate(self.shapelets_values.keys()):
            self._log("Transforming for dilation {} ({}/{}) with {} shapelets".format(
                dil, i, len(self.shapelets_values), len(self.shapelets_values[dil])))
            if len(self.shapelets_values[dil]) > 0:
                dilation = dil
                X_strides = self._get_X_strides(X, 9, dilation, 0)
                d = shapelet_dist_numpy(
                    X_strides, self.shapelets_values[dil])
                distances[:, prev:prev+d.shape[1]] = d
                prev += d.shape[1]
        return distances

    def _get_X_strides(self, X, length, dilation, padding):
        n_samples, _, n_timestamps = X.shape
        if padding > 0:
            X_pad = np.zeros((n_samples, n_timestamps+2*padding))
            X_pad[:, padding:-padding] = X[:, self.id_ft, :]
        else:
            X_pad = X[:, self.id_ft, :]
        X_strides = generate_strides_2D(X_pad, length, dilation)
        X_strides = (X_strides - X_strides.mean(axis=-1, keepdims=True)) / (
            X_strides.std(axis=-1, keepdims=True) + 1e-8)
        return X_strides

    def _get_regions(self, indexes):
        regions = []
        region = []
        for i in range(indexes.shape[0]-1):
            region.append(indexes[i])
            if indexes[i] != indexes[i+1]-1:
                regions.append(region)
                region = []
        if len(region) > 0:
            regions.append(region)
        return regions

    def _generate_inputs(self, X, y):
        """


        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        """
        self._log("Performing MiniRocket Transform")
        m = MiniRocket().fit(X)
        ft, locs = m.transform(X, return_locs=True)
        self._log(
            "Performing kernel selection with {} kernels".format(locs.shape[1]))
        if self.use_class_weights:
            rf = RandomForestClassifier(n_estimators=self.n_trees,
                                        max_features=self.max_ft,
                                        class_weight='balanced',
                                        ccp_alpha=0.01,
                                        n_jobs=self.n_threads)
        else:
            rf = RandomForestClassifier(n_estimators=self.n_trees,
                                        max_features=self.max_ft,
                                        n_jobs=self.n_threads)
        rf.fit(ft, y)
        dilations, num_features_per_dilation, biases = m.parameters
        dils = np.zeros(biases.shape[0], dtype=int)
        n = 0
        for i in range(dilations.shape[0]):
            dils[n:84*(num_features_per_dilation[i])+n] = dilations[i]
            n += 84*num_features_per_dilation[i]

        tree_splits = []

        def extract_tree_splits(tree, features,  y,):
            tree_ = tree.tree_
            x_id = np.asarray(range(features.shape[0]))

            def recurse(node, depth, x_id):
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    ft_id = tree_.feature[node]
                    threshold = tree_.threshold[node]
                    L = x_id[np.where(features[x_id, ft_id] <= threshold)[0]]
                    R = x_id[np.where(features[x_id, ft_id] > threshold)[0]]
                    y_node = np.zeros(x_id.shape[0], dtype=np.int32)
                    y_node[L.shape[0]:] += 1
                    recurse(tree_.children_left[node], depth + 1,
                            x_id[np.where(features[x_id, ft_id] <= threshold)[0]])
                    recurse(tree_.children_right[node], depth + 1,
                            x_id[np.where(features[x_id, ft_id] > threshold)[0]])
                    tree_splits.append(
                        [np.concatenate((L, R), axis=0), y_node, ft_id])
            recurse(0, 1, x_id)

        for dt in rf.estimators_:
            extract_tree_splits(dt, ft, y)

        self._log("Extracted {} splits".format(len(tree_splits)))
        return locs, dils, biases, tree_splits

    def _check_is_fitted(self):
        """


        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if any(self.__dict__[attribute] is None for attribute in ['shapelets_values']):
            raise AttributeError("CST is not fitted, call the fit method before"
                                 "attemping to transform data")

@njit(parallel=True, fastmath=True, cache=True)
def process_all_nodes(X, y, L, dil, P, X_indexes, Y_indexes, K_indexes):
    candidates = np.zeros((X_indexes.shape[0]*20,9),dtype=np.float32)
    m = 0
    for i_split in prange(X_indexes.shape[0]):        
    
        i_x = np.where(X_indexes[i_split]>=0)[0]
        x = X[X_indexes[i_split][i_x]]
        iy = Y_indexes[i_split][i_x]
        l = L[X_indexes[i_split][i_x], i_split]
        k =  K_indexes[i_split]
        c = _process_node(x, iy, l, dil, P)

    return candidates

@njit(cache=True)
def _generate_strides_2d(X, window_size, window_step):
    n_samples, n_timestamps = X.shape
    
    shape_new = (n_samples,
                 n_timestamps - (window_size-1)*window_step,
                 window_size // 1)
    s0, s1 = X.strides
    strides_new = (s0, s1, window_step *s1)
    return as_strided(X, shape=shape_new, strides=strides_new)

@njit(cache=True)
def _get_regions(indexes):
    regions = np.zeros((indexes.shape[0]*2),dtype=np.int64)-1
    p = 0
    for i in prange(indexes.shape[0]-1):
        regions[p] = indexes[i]
        if indexes[i] == indexes[i+1]-1:
            p+=1
        else:
            p+=2
    regions[p] = indexes[-1]
    idx = np.where(regions!=-1)[0]
    return regions[idx], np.concatenate((np.array([0],dtype=np.int64),np.where(np.diff(idx)!=1)[0]+1,np.array([indexes.shape[0]],dtype=np.int64)))

@njit(fastmath=True, cache=True)
def _process_node(X, y, L, dilation, P):
    classes = np.unique(y)
    n_classes = classes.shape[0]
    
    Lp = _generate_strides_2d(L, 9, dilation).sum(axis=-1)
    
    c_w = X.shape[0] / (n_classes * np.bincount(y))
    LC = np.zeros((n_classes, Lp.shape[1]),dtype=np.float32)
    for i_class in prange(n_classes):
        LC[i_class] = c_w[i_class] * Lp[
            np.where(y == i_class)[0]].sum(axis=0)
    
    candidates_grp = np.zeros((1,9),dtype=np.float32)
    
    for i_class in prange(n_classes):
        if LC.sum() > 0:
            D = LC[i_class] - LC[(i_class+1) % 2]
            id_D = np.where(D >= np.percentile(D, P))[0]
            regions, i_regions = _get_regions(id_D)
            for i_r in prange(i_regions.shape[0]-1):
                region = regions[i_regions[i_r]:i_regions[i_r+1]]
                LC_region = LC[i_class][region]
                id_max_region = region[LC_region.argmax()]
                x_index = np.argmax(
                    Lp[np.where(y == i_class)[0], id_max_region])
                candidate = np.zeros((1,9),dtype=np.float32)
                for j in prange(9):
                    candidate[0,j] = X[np.where(y == i_class)[0][x_index], 0, id_max_region+j*dilation]
                candidates_grp = np.concatenate((candidates_grp, candidate),axis=0)
    
    return candidates_grp
    
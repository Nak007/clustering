'''
Function (f)
------------
 1) update_params   --> (c1,c2)
 2) cluster_results
 3) elbow_plot      --> (f2)
 4) silhouette_plot --> (f2)
 5) gap_stat_plot   --> (f2)
 6) dendogram_plot  --> (f2)
 7) matplotlib_cmap --> (c3,c4,f9,f10,f11)
 8) create_cmap     --> (c3,c4,f9,f10,f11)
 9) cluster_pie
10) cluster_scatter   --> (c3)
11) cluster_histogram --> (c3)

Class (c)
---------
 1) cluster_kmeans
 2) cluster_linkage
 3) cluster_factors
 4) radar_plot
 
 Note: '--> (x)' : embedded in x 
'''
import pandas as pd, numpy as np, inspect

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import gaussian_kde
from scipy.stats.mstats import gmean
from scipy.ndimage.filters import gaussian_filter1d as smt
from scipy import stats

import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

def update_params(obj, param_dict):
    
    '''
    Parameters
    ----------
    obj : class or function object
    
    param_dict : dictionary of properties
    \t A dictionary to override the default properties
    \t of given class or function.
    
    Returns
    -------
    dictionary of updated properties 
    '''
    val = inspect.getfullargspec(obj)[3]
    arg = inspect.getfullargspec(obj)[0][-len(val):]
    params = dict([n for n in zip(arg,val)])
    keys = set(params.keys()).intersection(param_dict.keys())
    return {**params, **dict([(k,param_dict.get(k)) for k in keys])}
   
class cluster_kmeans:
  
    '''
    This class is using sklearn.cluster.Kmeans interface
    as well as its parameters
    
    Parameters
    ----------
    clusters : tuple of int, optional, (default:(1,10))
    \t Beginning and finishing number of clusters 
    
    frac : float, optional, (default:0.5)
    \t Fraction of axis items to return (sample size).
    \t This only applies to creating bootstrapped dataset.
    
    n_bootstrap : int, optional, (default:5)
    \t Number of bootstrappings for Gap-statistics
    
    p_samples : float, optional, (default:0.5)
    \t Percent of samples that is used to calculate 
    \t silhouette coefficient. It is calculated using the 
    \t mean intra-cluster distance (a) and the mean 
    \t nearest-cluster distance (b) for each sample. 
    \t The formula (for a sample) is (b - a) / max(a, b).
    \t see https://scikit-learn.org for more info
    
    **params : KMeans properties, optional
    \t params are used to specify or override properties 
    \t of sklearn.cluster.KMeans such as n_clusters, init.
    \t see https://scikit-learn.org for more info
    '''
    def __init__(self, clusters=(1,10), frac=0.5, n_bootstraps=5, p_samples=0.5, **params):
        
        self.params = update_params(KMeans,params)
        self.clusters = np.arange(max(clusters[0],1),clusters[1]+1) 
        self.frac = frac
        self.n_bootstraps = n_bootstraps
        self.p_samples = p_samples

    def fit(self, X):
        
        '''
        Parameters
        ----------
        X : dataframe object, of shape (n_samples, n_features)
        \t Training data. All elements of X must be finite, 
        \t i.e. no NaNs or infs.
        
        Returns
        -------
        self.data : dictionary of 
        - n_clusters : an array of nth clusters
        - wcss : a list of within-cluster sum-of-squares
        - silhouette : a list of silhouette coefficients
        - gap_stat : a list of gap-statistics
        - labels : an array of shape (n_samples, n_clusters)
          it contains labels (cluster) for each point from each
          ith iteration.
          
        Note: 
        For Gap Statistic, scipy.stats.mstats.gmean is used to
        calculate geometric means of "wcss" from expected null 
        distributions. This function can handle product of many
        massive numbers as opposed to numpy.prod().
        '''
        # k is the number of clusters
        k = len(self.clusters)
        data = dict(n_clusters=self.clusters, wcss=[None]*k, 
                    silhouette=[None]*k, gap_stat=[None]*k, labels=[None]*k)
        if self.p_samples != None: sample_size = int(self.p_samples*len(X))
        else: sample_size = None
                 
        # Keyword arguments for silhouette_score
        kwargs = dict(metric='euclidean', sample_size=sample_size, 
                      random_state=self.params['random_state'])
        
        # Creating bootstrapped dataset(s)
        np.random.seed(self.params['random_state'])
        rand_states = np.random.randint(0,100,size=self.n_bootstraps)
        bootstrap = [X.sample(frac=self.frac, replace=True, 
                              random_state=n) for n in rand_states]
        
        for (n,k) in enumerate(self.clusters): 
            self.params['n_clusters'] = k
            model = KMeans(**self.params)
            model.fit(X)
            data['wcss'][n] = model.inertia_
            data['labels'][n] = model.labels_.reshape(-1,1)
            if k==1: score = 0
            else: score = silhouette_score(X, model.labels_, **kwargs)
            data['silhouette'][n] = score
            data['gap_stat'][n] = self.__gap_statistics(model,bootstrap)
        data['labels'] = np.hstack(data['labels'])
        self.data = data.copy()
        del data, bootstrap
        
    def __gap_statistics(self, estimator, bootstrap):
        
        inertia = estimator.inertia_
        wcss = [None] * len(bootstrap)
        for n,X in enumerate(bootstrap):
            estimator.fit(X)
            wcss[n] = estimator.inertia_
        return np.log(gmean(wcss)) - np.log(inertia)
       
class cluster_linkage: 
    
    '''
    This class performs hierarchical clustering bu using 
    'linkage' and 'fcluster' from scipy.cluster.hierarchy 
    to determine linkage among records.
    
    Parameters
    ----------
    method : str, optional, (defualt:'ward')
    \t The linkage algorithm to use. Here is the list 
    \t of methods can be chosen from i.e. 'single',
    \t 'complete', 'average', 'weighted', 'centroid',
    \t 'median', 'ward'

    metric : str or function, optional
    \t The distance metric to use in the case that X is 
    \t a collection of observation vectors. 
    
    n_clusters : int, optional, (default: 10)
    \t It defines number of clusters for algorithm to 
    \t determine clustering metrics such as silhouette
    
    p_samples : float, optional, (default:0.5)
    \t Percent of samples that is used to calculate 
    \t silhouette coefficient. It is calculated using the 
    \t mean intra-cluster distance (a) and the mean 
    \t nearest-cluster distance (b) for each sample. 
    \t The formula (for a sample) is (b - a) / max(a, b).
    \t see https://scikit-learn.org for more info
    
    **params : KMeans properties, optional
    \t params are used to specify or override properties 
    \t of sklearn.cluster.KMeans such as n_clusters, init.
    \t In this class, KMeans is used to only compute
    \t "within cluster sum of sqaures" or "wcss".
    \t see https://scikit-learn.org for more info
    '''
    def __init__(self, method='ward', metric='euclidean', n_clusters=10, p_samples=0.5, **params):
        
        self.params = update_params(KMeans,params)
        self.kwargs = dict(method=method, metric=metric)
        self.n_clusters = np.arange(1,n_clusters+1)
        self.p_samples = p_samples
      
    def fit(self, X):
        
        '''
        Parameters
        ----------
        X : dataframe object, of shape (n_samples, n_features)
        \t Training data. All elements of X must be finite, 
        \t i.e. no NaNs or infs.
        
        Returns
        -------
        self.data : dictionary of 
        - n_clusters : an array of nth clusters
        - wcss : a list of within-cluster sum-of-squares
        - silhouette : a list of silhouette coefficients
        - labels : an array of shape (n_samples, n_clusters)
          it contains labels (cluster) for each point from each
          ith iteration.
        - Z : The hierarchical clustering encoded as a linkage 
          matrix. A(n−1) by 4 matrix Z is returned.
        '''
        k = len(self.n_clusters)
        data = dict(n_clusters=self.n_clusters, wcss=[None]*k, 
                    silhouette=[None]*k, labels=[None]*k)
        if self.p_samples != None: sample_size = int(self.p_samples*len(X))
        else: sample_size = None
        
        # Keyword arguments for silhouette_score
        kwargs = dict(metric='euclidean', sample_size=sample_size, 
                      random_state=self.params['random_state'])
          
        # Linkage matrix
        Z = linkage(X, **self.kwargs); data['Z'] = Z
        data['dist'] = Z[-k:,2][::-1]
        
        for n,k in enumerate(self.n_clusters):
            
            labels = fcluster(Z, k, criterion='maxclust')
            data['labels'][n] = labels.reshape(-1,1)
            
            self.params['n_clusters'] = k
            model = KMeans(**self.params)
            model.fit(X,labels)
            data['wcss'][n] = model.inertia_
            
            if k==1: score = 0
            else: score = silhouette_score(X, labels, **kwargs)
            data['silhouette'][n] = score
            
        data['labels'] = np.hstack(data['labels'])
        self.data = data.copy()
        del data
        

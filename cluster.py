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
        
def elbow_plot(axis, wcss, n_clusters=None):

    '''
    ** Elbow method **
     
    This method looks at change of "within-cluster-sum
    of-squared" or so-called "wcss". One should choose a 
    number of clusters so that adding another cluster 
    doesn't reduce significant amount of erros. More 
    precisely, if one plots wcss by the clusters against 
    the number of clusters, one should choose the point
    before marginal gain becomes minute or insignificant.

    Parameters
    ----------
    axis : matplotlib axis object
    \t axis refers to a single Axes object

    wcss : array of float
    \t list of Within-Cluster-Sum-of-Squareds.
    
    n_clusters : list of int, optional, (defualt: None)
    \t list of nth clusters. If None, list of 1 to n 
    \t clusters is used instead, where n indicates 
    \t number of items in "wcss".
    '''
    labels = ['Sum of WCSSs','Rate of Change','Optimal Number of Clusters']
    x = np.arange(len(wcss))
    if n_clusters is None: n_clusters = np.arange(1,len(wcss)+1)
    kwargs = dict(lw=1.5, marker='o', markersize=4, color='#0652DD')
    line1 = axis.plot(x, wcss, **kwargs)

    kwargs = dict(color='#3d3d3d', fontsize=10, fontweight='bold')
    axis.set_xlabel('Number of clusters', **kwargs)
    axis.set_ylabel('Within Culster\nSum of Squared (WCSS)', **kwargs)
    axis.set_facecolor('white')
    axis.set_title('Clustering: Elbow method', fontsize=14)
    axis.set_xticks(x)
    axis.set_xticklabels(n_clusters, color='#3d3d3d', fontsize=10)
    axis.grid(False, color='#aaa69d', lw=0.5, ls='--')
    
    # set y_lim within 5% of max - min
    incr = (max(wcss) - min(wcss))*0.05
    axis.set_ylim(min(wcss)-incr,max(wcss)+incr)
    axis.set_xlim(-0.5,len(x)-0.5)

    # 2nd derivative of the distances
    d = np.diff(wcss)/wcss[:-1]
    c = [np.nan] + (np.diff(d)/abs(d[:-1])).tolist() + [np.nan]
   
    # Rate of Change axis
    tw_axis = axis.twinx()
    kwargs = dict(lw=1, marker='o',markersize=4, ls='--', color='#ff5252')
    line2 = tw_axis.plot(x, c, **kwargs)
    kwargs = dict(color='#3d3d3d', fontsize=11, fontweight='bold')
    tw_axis.set_ylabel('Rate of change', **kwargs)
    ylim = tw_axis.get_ylim()
    tw_axis.set_ylim(ylim[0],ylim[1]*3/2)

    # General selection criterion
    n = np.nanargmax(c)
    span = axis.axvspan(-0.5, n, color='#aaa69d', alpha=0.2)
    axis.axvline(n, color='#3d3d3d', ls='--', lw=0.8)
    axis.legend([line1[0], line2[0], span], labels, loc='best', fontsize=10)
    
def silhouette_plot(axis, silhouette, n_clusters=None):
    
    '''
    ** Silhouette Coefficient **
    
    It is calculated using the mean intra-cluster 
    distance (a) and the mean nearest-cluster distance 
    (b) for each sample. The formula (for a sample) is 
    (b - a) / max(a, b)

    Parameters
    ----------
    axis : matplotlib axis object
    \t axis refers to a single Axes object

    silhouette : array of float
    \t list of silhouette coefficients.

    n_clusters : list of int, optional, (defualt: None)
    \t list of nth clusters. If None, list of 1 to n 
    \t clusters is used instead, where n indicates 
    \t number of items in "silhouette".
    '''
    labels = ['Avg. Silhouette','Optimal Number of Clusters']
    x = np.arange(len(silhouette))
    if n_clusters is None: n_clusters = np.arange(1,len(silhouette)+1)
    kwargs = dict(lw=1.5, marker='o',markersize=4, color='#0652DD')
    line1 = axis.plot(x, silhouette, **kwargs)
    
    kwargs = dict(color='#3d3d3d', fontsize=11, fontweight='bold')
    axis.set_xlabel('Number of clusters', **kwargs)
    axis.set_ylabel('Average silhouette score', **kwargs)
    axis.set_facecolor('white')
    axis.set_title('Clustering: Silhouette method',fontsize=14)
    axis.set_xticks(x)
    axis.set_xticklabels(n_clusters, color='#3d3d3d', fontsize=10)
    
    # General selection criterion
    n = np.nanargmax(silhouette)
    line2 = axis.axvline(n, color='#3d3d3d', ls='--', lw=0.8)
    axis.legend([line1[0], line2], labels, loc='best', fontsize=10)
    
    # set y_lim within 5% of max - min
    incr = (max(silhouette) - min(silhouette))*0.05
    axis.set_ylim(min(silhouette)-incr,max(silhouette)+incr)
    axis.set_xlim(-0.5,len(x)-0.5)
    
    axis.axhspan(0.25, 0.50, color='#aaa69d', alpha=0.2, hatch='//')
    axis.axhspan(0.50, 0.70, color='#70a1ff', alpha=0.3, hatch='//')
    axis.axhspan(0.70, 1.00, color='#2ed573', alpha=0.4, hatch='//')
    kwargs = dict(fontsize=10, color='#3d3d3d', va='bottom', ha='left')
    axis.text(-0.3, 0.26, r'Artificial $(\geq0.25)$', **kwargs)
    axis.text(-0.3, 0.51, 'Reasonable $(\geq0.50)$', **kwargs)
    axis.text(-0.3, 0.71, 'Strong $(\geq0.70)$', **kwargs)
    kwargs = dict(color='#3d3d3d', lw=0.5, ls='--')
    axis.axhline(0.25, **kwargs)
    axis.axhline(0.50, **kwargs)
    axis.axhline(0.70, **kwargs)
    incr = (1 - min(silhouette))*0.05
    axis.set_ylim(min(silhouette)-incr,1)

def gap_stat_plot(axis, gap_stat, n_clusters=None):
    
    '''
    ** Gap Statistic **
    
    The idea of the Gap statistic is to compare the wcss 
    (dispersion) to its expectation under an appropriate 
    null reference distribution e.g. bootstrapped data.
    It can be mathematically expressed as:
    
            Gap(k) = Log(E[wcss(k,n)]) - Log(wcss)
            
    where k and n respresent number of clusters and
    number of bootstrappings, respectively.

    Parameters
    ----------
    axis : matplotlib axis object
    \t axis refers to a single Axes object

    gap_stat : array of float
    \t list of Gap Statistics.

    n_clusters : list of int, optional, (defualt: None)
    \t list of nth clusters. If None, list of 1 to n 
    \t clusters is used instead, where n indicates 
    \t number of items in "gap_stat".
    '''
    labels = ['Gap Statistic','Optimal Number of Clusters']
    x = np.arange(len(gap_stat))
    if n_clusters is None: n_clusters = np.arange(1,len(gap_stat)+1)
    if n_clusters[0]==1: gap_stat[0] = np.nan
    kwargs = dict(lw=1.5, marker='o',markersize=4, color='#0652DD')
    line = axis.plot(x, gap_stat, **kwargs)
    
    kwargs = dict(color='#3d3d3d', fontsize=11, fontweight='bold')
    axis.set_xlabel('Number of clusters', **kwargs)
    axis.set_ylabel(r'Log(E[WCSS]) - Log(WCSS)', **kwargs)
    axis.set_facecolor('white')
    axis.set_title('Clustering: Gap Statistics',fontsize=14)
    axis.set_xticks(x)
    axis.set_xticklabels(n_clusters, color='#3d3d3d', fontsize=10)
    axis.grid(True, color='#aaa69d', lw=0.5, ls='--')
    
    # set y_lim within 5% of max - min
    incr = (np.nanmax(gap_stat) - np.nanmin(gap_stat))*0.05
    axis.set_ylim(np.nanmin(gap_stat)-incr,np.nanmax(gap_stat)+incr)
    axis.set_xlim(-0.5,len(x)-0.5)

    # General selection criterion
    n = np.nanargmax(gap_stat)
    span = axis.axvspan(-0.5, n, color='#aaa69d', alpha=0.2)
    axis.axvline(n, color='#3d3d3d', ls='--', lw=0.8)
    axis.legend([line[0], span], labels, loc='best', fontsize=10)
    
def dendogram_plot(axis, Z, p=10):
    
    '''
    Parameters
    ----------
    axis : matplotlib axis object
    \t axis refers to a single Axes object
    
    Z : array of floats, of shape (n_samples,4)
    \t The linkage matrix encoding the hierarchical 
    \t clustering to render as a dendrogram
    
    p : int, optional, (default:10)
    \t The p parameter for truncate_mode
    
    Note:
    see https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.cluster.hierarchy.dendrogram.html
    '''
    kwargs = dict(p=p, truncate_mode='lastp', leaf_rotation=90, 
                  leaf_font_size=10, show_contracted=True, ax=axis)
    dendro = dendrogram(Z, **kwargs)

    axis.set_title('Clustering: Dendrogram', fontsize=14)
    kwargs = dict(color='#3d3d3d', fontsize=11, fontweight='bold')
    axis.set_xlabel('Cluster size', **kwargs)
    axis.set_ylabel('Distance', **kwargs)
    axis.set_facecolor('White')
    axis.grid(False)
    
    kwargs = dict(xytext=(0,-5), textcoords='offset points', 
                  va='top', ha='center', fontsize=10)
    # icoord: index coordinates, dccord: distance coordinates
    for i, d, c in zip(dendro['icoord'],dendro['dcoord'],
                       dendro['color_list']):
        x, y = 0.5 * sum(i[1:3]), d[1]
        axis.plot(x, y, marker='o',markersize=4, c=c)
        axis.annotate("%.3g" % y, (x, y), **kwargs)
        
def cluster_results(data, transform='normalize', n_columns=2, figsize=(5.5,3.5), fname=None):
    
    '''
    Parameters
    ----------
    data : dictionary
    \t data that is obtained from 'cluster_kmeans' or
    \t 'cluster_linkage' only.
    
    transform : str, optional, (default: 'log') 
    \t This only applies to wcss whether to 
    \t - 'normalize' : divide wcss by max(wcss)
    \t - 'log' : take natural log of (wcss+1)
    \t - 'none' : use actul wcss values
    
    n_columns : int, optional, (default:2)
    \t n_columns represents number of columns in figure.
    
    figsize : (float, float), optional, (default:(5.5,3.5))
    \t width, height in inches for individual plot
    
    fname : str or PathLike or file-like object
    \t A path, or a Python file-like object
    '''
    # Find number of plots and layout
    plots = ['wcss','silhouette','gap_stat','Z']
    plots = np.unique(list(set(plots).intersection(data.keys())))
    n_plots = len(plots)
    n_rows = np.ceil(n_plots/n_columns).astype(int)
    shape = (n_rows,n_columns)
    loc = [(r,c) for c in range(n_columns) 
           for r in range(n_rows)][:n_plots]
       
    f = (figsize[0]*n_columns,figsize[1]*n_rows)
    fig = plt.subplots(figsize=f)
    axes = [plt.subplot2grid(shape,l) for l in loc]
    for (n,p) in enumerate(plots):
        y = np.array(data[p].copy())
        if p=='wcss':
            if transform=='log': y = np.log(y+1)
            elif transform=='normalize': y = y/max(y)
            elbow_plot(axes[n], y)
        elif p=='silhouette': silhouette_plot(axes[n], y)
        elif p=='gap_stat': gap_stat_plot(axes[n], y)
        elif p=='Z': dendogram_plot(axes[n], y)
    plt.tight_layout()
    if fname is not None: plt.savefig(fname)
    plt.show()

def matplotlib_cmap(name='viridis', n=10):

    '''
    Parameters
    ----------
    name : matplotlib.colors.Colormap or str or None, 
    optional, (default:'viridis')
    \t the name of a colormap known to Matplotlib. 
    
    n : int, optional, (defualt: 10)
    \t Number of shades for defined color map
    
    Returns
    -------
    List of color-hex codes from defined 
    matplotlib.colors.Colormap. Such list contains
    "n" shades.
    '''
    c_hex = '#%02x%02x%02x'
    c = cm.get_cmap(name)(np.linspace(0,1,n))
    c = (c*255).astype(int)[:,:3]
    return [c_hex % (c[i,0],c[i,1],c[i,2]) for i in range(n)]
  
def create_cmap(c1=(23,10,8), c2=(255,255,255)):
    
    '''
    Creating matplotlib.colors.Colormap (Colormaps)
    with two colors
    
    Parameters
    ----------
    c1 : hex code or (r,g,b), optional, 
    default:(23,10,8)
    \t The beginning color code
    
    c2 : hex code or (r,g,b), optional,
    default:(170,166,157)
    \t The ending color code
    
    Returns
    -------
    matplotlib.colors.ListedColormap
    '''
    def to_rgb(c):
        c = c.lstrip('#')
        return tuple(int(c[i:i+2],16) for i in (0,2,4))
    # Convert to RGB
    if isinstance(c1,str): c1 = to_rgb(c1)
    if isinstance(c2,str): c2 = to_rgb(c2)
    colors = np.ones((256,4))
    for i in range(3):
        colors[:,i] = np.linspace(c1[i]/256,c2[i]/256,256)
    colors = colors[np.arange(255,-1,-1),:]
    return ListedColormap(colors)

def cluster_pie(axis, y, colors=None, labels=None):
    
    '''
    Parameters
    ----------
    axis : matplotlib axis object
    \t axis refers to a single Axes object
    
    y : 1D-array or pandas.core.series.Series
    \t Array of cluster labels (0 to n_clusters-1)
    
    colors : list of color-hex codes or (r,g,b), 
    optional, (default:None)
    \t List must contain at the very least, 'n' of 
    \t color-hex codes or (r,g,b) that matches number of 
    \t clusters in 'y'. If None, the matplotlib color maps, 
    \t namely 'gist_rainbow' is used.
    
    labels : List of str, optional, (defualt:None)
    \t List of labels (integer) whose items must be arranged
    \t in ascending order. If None, (n+1) cluster is 
    \t assigned, where n in the cluster label.
    '''
    a, sizes = np.unique(y,return_counts=True)
    if colors is None: 
        colors = matplotlib_cmap('gist_rainbow',len(sizes))
    explode = (sizes==max(sizes)).astype(int)*0.1
    if labels is None: 
        labels = ['Cluster %d \n (%s)' % (m+1,'{:,d}'.format(n)) 
                  for m,n in zip(a,sizes)]
    else: labels = ['%s \n (%s)' % (m,'{:,d}'.format(n)) 
                    for m,n in zip(labels,sizes)]
    kwargs = dict(explode=explode, labels=labels, autopct='%1.1f%%', 
                  shadow=True, startangle=90, colors=colors, 
                  wedgeprops = dict(edgecolor='black'))
    axis.pie(sizes, **kwargs)
    axis.axis('equal')
    
def cluster_scatter(axis, x1, x2, y, colors=None, alpha=0, labels=None):
    
    '''
    Parameters
    ----------
    axis : matplotlib axis object
    \t axis refers to a single Axes object
    
    x1 : pandas.core.series.Series
    \t X-axis variable. All elements of X must be finite, 
    \t i.e. no NaNs or infs. 
    
    x2 : pandas.core.series.Series
    \t Y-axis variable. All elements of X must be finite, 
    \t i.e. no NaNs or infs.
    
    y : 1D-array or pandas.core.series.Series
    \t Array of cluster labels (0 to n_clusters-1)
    
    colors : list of color-hex codes or (r,g,b), 
    optional, (default:None)
    \t List must contain at the very least, 'n' of 
    \t color-hex codes or (r,g,b) that matches number of 
    \t clusters in 'y'. If None, the matplotlib color maps, 
    \t namely 'gist_rainbow' is used.
    
    alpha : float, optional, (default: 0)
    \t 'alpha' refers to the probability outside the  
    \t confidence such interval is ignored in the plot
    
    labels : List of str or str, optional, (defualt:None)
    \t List of labels whose items must be arranged in
    \t ascending order. If 'auto', 'Cluster {n+1}' is 
    \t assigned, where n in the cluster label. 
    \t If 'centroid', centroid of cluster is plotted.
    \t If None, no labels are displayed
    '''
    n_labels = len(np.unique(y))
    bbox = dict(boxstyle="circle", fc='w', ec='k', pad=0.5)
    props = dict(fontweight='bold', color='k', va='center',ha='center', 
                 fontsize=11, bbox=bbox)
    
    if colors is None: colors = matplotlib_cmap('gist_rainbow',n_labels)
    text_xy, plot = [None] * n_labels, [None] * n_labels
        
    for n,k in enumerate(np.unique(y)):
        X = [x1[y==k],x2[y==k]]
        xx = np.vstack([np.array(n).ravel() for n in X])
        cmap = create_cmap(c1=colors[k],c2=(255,255,255))
        kwargs = dict(c=gaussian_kde(xx)(xx), s=10, alpha=0.8, cmap=cmap) 
        axis.scatter(xx[0], xx[1], **kwargs)
        kwargs = dict(c=colors[k], s=10, alpha=1) 
        plot[n] = axis.scatter(X[0].mean(), X[1].mean(), **kwargs)
        text_xy[n] = (X[0].mean(),X[1].mean(),str(k+1))
        
    kwargs = dict(color='#3d3d3d', fontsize=10, fontweight='bold')
    axis.set_xlabel('%s' % x1.name, **kwargs)
    axis.set_ylabel('%s' % x2.name, **kwargs)
    axis.set_facecolor('white')
    
    axis.set_xlim(tuple(np.nanpercentile(x1,[alpha,100-alpha])))
    axis.set_ylim(tuple(np.nanpercentile(x2,[alpha,100-alpha])))
    
    # legend and labels
    if (labels=='auto') | (isinstance(labels,list)): 
        if labels=='auto': 
            labels = ['Cluster {0}'.format(n+1) for n in np.unique(y)]
        axis.legend(plot, labels, loc='best', fontsize=8, framealpha=0)
    elif labels=='centroid':
        for n in text_xy:
            axis.text(n[0], n[1], n[2], **props)
    
def cluster_histogram(axis, x, y, colors=None, bins=100, sigma=1, labels=None):
    
    '''
    Parameters
    ----------
    axis : matplotlib axis object
    \t axis refers to a single Axes object
    
    x : pandas.core.series.Series
    \t Y-axis variable. All elements of X must be finite, 
    \t i.e. no NaNs or infs.
    
    y : 1D-array or pandas.core.series.Series
    \t Array of cluster labels (0 to n_clusters-1)
    
    colors : list of color-hex codes or (r,g,b), 
    optional, (default:None)
    \t List must contain at the very least, 'n' of 
    \t color-hex codes or (r,g,b) that matches number of 
    \t clusters in 'y'. If None, the matplotlib color maps, 
    \t namely 'gist_rainbow' is used.
    
    bins : int, optional, (default:100)
    \t Number of bins to determine PDF
    
    sigma : float, optional, (default:1)
    \t Standard deviation for Gaussian kernel. Sigma must 
    \t be greater than 0. The higher the sigma the smoother
    \t the probability density curve (PDF)
    
    labels : List of str or str, optional, (defualt:None)
    \t List of labels whose items must be arranged in
    \t ascending order. If 'auto', 'Cluster {n+1}' is 
    \t assigned, where n in the cluster label. If 'mle', 
    \t Maximum Likelihood Estimation (mle) is performed and 
    \t located in PDF. If None, no labels are displayed
    '''
    n_labels = len(np.unique(y))
    min_x, max_x = '%#1.4g' % min(x), '%#1.4g' % max(x)
    width = '%#1.3g' % ((max(x)-min(x))/bins)
    s = '(min:{0}, max:{1}, width:{2})'.format(min_x, max_x, width)
    bbox = dict(boxstyle="circle", fc='w', ec='k', pad=0.5)
    props = dict(fontweight='bold', color='k', va='bottom', 
                 ha='center', fontsize=11, bbox=bbox)
    
    if colors is None: colors = matplotlib_cmap('gist_rainbow',n_labels)
    bins = np.histogram(x,bins=bins)[1]; ticks = np.arange(len(bins)-1)
    text_xy, plot = [None] * n_labels, [None] * n_labels
    
    for n,k in enumerate(np.unique(y)):
        hist = np.histogram(x[y==k], bins=bins, density=True)[0]
        pdf = smt(hist,sigma); pdf = pdf/sum(pdf)
        plot[n] = axis.fill_between(ticks, pdf, color=colors[k], alpha=0.5)
        axis.plot(ticks, pdf, color=colors[k], lw=1)
        text_xy[n] = (np.argmax(pdf),max(pdf),str(k+1))
    
    kwargs = dict(color='#3d3d3d', fontsize=10, fontweight='bold')
    axis.set_title(x.name, **kwargs)
    axis.set_xlabel(s, **kwargs)
    axis.set_ylabel('Density', **kwargs)
    axis.set_facecolor('white')
    y_max = axis.get_ylim()[1]*5/3
    axis.set_ylim(0,y_max)
    
    # legend and labels
    if (labels=='auto') | (isinstance(labels,list)): 
        if labels=='auto': 
            labels = ['Cluster {0}'.format(n+1) for n in np.unique(y)]
        axis.legend(plot, labels, loc='best', fontsize=8, framealpha=0)
    elif labels=='mle':
        for n in text_xy:
            axis.text(n[0], n[1]+y_max*0.05, n[2], **props)
        
class cluster_factors:
    
    '''
    This function plots variables given set of labels.
    The type of plots are 'scatter' and 'hist'.
    
    Parameters
    ----------
    random_state : int, optional, (defualt:0)
    \t Seed for the random number generator
    
    frac : float, optional, (default:0.5)
    \t Percentage of samples to be randomly selected.
    \t Fraction of axis items to return. 
    
    n_columns : int, optional, (default: 4)
    \t Number of display columns
    
    figsize : (float, float), optional, (default:(4,3))
    \t width, height in inches
    '''
    def __init__(self, random_state=0, frac=0.5, n_columns=4, figsize=(4,3)):

        self.kwargs = dict(random_state=0, frac=0.5, replace=False)
        self.n_columns = n_columns
        self.figsize = figsize
  
    def fit(self, X, y, factor=None, method='scatter', alpha=0, colors=None, 
            bins=100, sigma=1, labels=True, fname=None):

        '''
        Parameters
        ----------
        X : dataframe object, of shape (n_samples, n_features)
        \t Sample data
        
        factor : str, optional, (default:None)
        \t Name of factor that is used for comparison against
        \t the rest of factors. If not defined, the first one
        \t from dataframe is selected. Only applicable for
        \t 'scatter'.
        
        method : str, optional, (default:'scatter')
        \t method of visualizations i.e. 'scatter' or 'hist'
        
        alpha : float, optional, (default: 0)
        \t 'alpha' refers to the probability outside the  
        \t confidence such interval is ignored in the plot
        
        colors : list of color-hex codes or (r,g,b), 
        optional, (default:None)
        \t List must contain at the very least, 'n' of 
        \t color-hex codes or (r,g,b) that matches number of 
        \t clusters in 'y'. If None, the matplotlib color maps, 
        \t namely 'gist_rainbow' is used.
        
        bins : int, optional, (default:100)
        \t Number of bins to determine PDF

        sigma : float, optional, (default:1)
        \t Standard deviation for Gaussian kernel. Sigma must 
        \t be greater than 0. The higher the sigma the smoother
        \t the probability density curve (PDF)
        
        labels : List of str or str, optional, (defualt:None)
        \t List of labels whose items must be arranged in
        \t ascending order. If 'auto', 'Cluster {n+1}' is 
        \t assigned, where n in the cluster label. If 'mle', 
        \t Maximum Likelihood Estimation (mle) is performed and 
        \t located in PDF. If None, no labels are displayed
    
        fname : str or PathLike or file-like object (default:None)
        \t A path, or a Python file-like object 
        \t (more info see matplotlib.pyplot.savefig)
        '''
        # random sampling
        XY = X.copy(); XY['labels'] = y
        XY = XY.sample(**self.kwargs)
        sample = XY.drop(columns=['labels']).reset_index(drop=True)
        labels_ = XY['labels'].reset_index(drop=True)
            
        # plot layout
        if method=='scatter': n_plots = X.shape[1]-1
        else: n_plots = X.shape[1]
        n_rows = int(np.ceil(n_plots/self.n_columns))
        shape = (n_rows, self.n_columns)
        loc = [(n,m) for n in range(n_rows) for m in 
               range(self.n_columns)][:n_plots]
        
        if (factor is None) & (method=='scatter'): 
            factor = list(X)[0]
            x1 = sample[factor]
        else: factor = None  
        columns = [n for n in list(X) if n != factor]
        
        figsize = (self.figsize[0]*self.n_columns,
                   self.figsize[1]*n_rows)
        fig = plt.figure(figsize=figsize)
        for (n,m) in enumerate(loc):
            axis = plt.subplot2grid(shape, m)
            x2 = sample[columns[n]]
            if method=='scatter':
                cluster_scatter(axis,x1,x2,labels_,colors,alpha,labels)
            elif method=='hist':
                cluster_histogram(axis,x2,labels_,colors,bins,sigma,labels)
        fig.tight_layout()
        if fname!=None: plt.savefig(fname)
        plt.show()
        
class radar_plot:
    
    '''
    Parameters
    ----------
    **params : dictionary of properties, optional
    \t params are used to specify or override properties 
    \t of the following functions, which are
    \t - axis.plot = {'ls':'-', 'lw':1, 'ms':3, 'marker':'o', 'fillstyle'='full'}
    \t - axis.fill = {'alpha'=0.5}
    \t - figure    = {'figsize':(6.4,4.8)}
    \t - pd.DataFrame.sample = {'random_state'=0, 'frac'=0.5)}
    '''
    def __init__(self, **params):
    
        init = dict(d0 = dict(ls='-', lw=1, ms=3, marker='o', fillstyle='full'),
                    d1 = dict(alpha=0.5), 
                    d2 = dict(figsize=(6.4,4.8)),
                    d3 = dict(random_state=0, frac=0.5))
        for k in init.keys():
            keys = set(init[k].keys()).intersection(params)
            init[k] = {**init[k], **dict((n,params[n]) for n in keys)}
        self.params = init
        
    def fit(self, X, y, q=50, title=None, colors=None, labels=None, fname=None):
        
        '''
        Parameters
        ----------
        X : dataframe object, of shape (n_samples, n_features)
        \t Training data. All elements of X must be finite, 
        \t i.e. no NaNs or infs.
        
        y : 1D-array or pandas.core.series.Series
        \t Array of cluster labels (0 to n_clusters-1)
        
        q : float, optional, (default:50)
        \t Percentile to compute, which must be between 0 and 
        \t 100 inclusive.

        title : str, optional, (default:None)
        \t Title of radar plot. If None, no title is displayed
        
        colors : list of color-hex codes or (r,g,b), 
        optional, (default:None)
        \t List must contain at the very least, 'n' of 
        \t color-hex codes or (r,g,b) that matches number of 
        \t clusters in 'y'. If None, the matplotlib color maps, 
        \t namely 'gist_rainbow' is used.
        
        labels : List of str, optional, (defualt:None)
        \t List of labels (integer) whose items must be arranged
        \t in ascending order. If None, (n+1) cluster is 
        \t assigned, where n in the cluster label.
        
        fname : str or PathLike or file-like object (default:None)
        \t A path, or a Python file-like object 
        \t (more info see matplotlib.pyplot.savefig)
        '''
        fig = plt.figure(**self.params['d2'])
        axis = plt.subplot(polar=True)
        self.__plot(axis, X, y, q, colors, labels)
        kwargs = dict(color='#3d3d3d', fontsize=12, fontweight='bold')
        axis.set_title(title, **kwargs)
        fig.tight_layout()
        if fname!=None: plt.savefig(fname)
        plt.show()

    def normalize(self, X, y, q=50):
        
        # resampling given random_state and frac
        features = X.columns
        XY = X.copy(); XY['y'] = y
        XY = XY.sample(**self.params['d3'])
        sample = XY.drop(columns=['y']).reset_index(drop=True)
        labels = XY['y'].reset_index(drop=True)
        del XY
        
        # normalize sample
        n_max = np.nanmax(sample, axis=0)
        n_min = np.nanmin(sample, axis=0)
        divisor = np.where((n_max-n_min)==0,1,n_max-n_min)
        norm_X = np.array((sample-n_min)/divisor)

        # percentile
        clusters = np.unique(labels)
        a = [np.nanpercentile(norm_X[labels==k],q,axis=0).ravel() 
             for k in clusters]
        return np.vstack(a), clusters, features
    
    def __plot(self, axis, X, y, q=50, colors=None, labels=None):
        
        # resample and normalize X
        X, y, c = self.normalize(X, y, q)
        if colors is None: colors = matplotlib_cmap('coolwarm_r',len(y))
        
        # angle of plots
        angles = [n/float(X.shape[1])*2*np.pi for n in range(X.shape[1])]
        angles += angles[:1]

        # if you want the first axis to be on top
        axis.set_theta_offset(np.pi/2)
        axis.set_theta_direction(-1)
        axis.set_rlabel_position(0)

        # draw one axe per variable + add labels 
        kwargs = dict(color='#3d3d3d', fontsize=10)
        axis.set_xticks(angles[:-1])
        axis.set_xticklabels(c, **kwargs)
        
        # set alignment of ticks
        for n,t in enumerate(axis.get_xticklabels()):
            if (0<angles[n]<np.pi): t._horizontalalignment = 'left'
            elif (angles[n]>np.pi): t._horizontalalignment = 'right'
            else: t._horizontalalignment = 'center'
        
        if labels is None: labels = ['Cluster {0}'.format(n+1) for n in y]
        for n,k in enumerate(y):
            values = X[k,:].tolist() + [X[k,0]]
            kwargs = dict(c=colors[n])
            axis.plot(angles, values, **{**self.params['d0'],**kwargs})
            kwargs = dict(c=colors[n], label=labels[n])
            axis.fill(angles, values, **{**self.params['d1'],**kwargs})
        
        axis.set_yticklabels([])
        kwargs = dict(facecolor='none', edgecolor='none',
                      fontsize=10, bbox_to_anchor=(0,0))
        axis.legend(**kwargs)
        axis.set_facecolor('White')
        axis.grid(True, color='grey', lw=0.5, ls='--') 
     
def local_outlier(X, **params):
    
    '''
    ** Local Outlier Factor (LOF) **
    
    The anomaly score of each sample is called Local 
    Outlier Factor. It measures the local deviation of 
    density of a given sample with respect to its 
    neighbors. It is local in that the anomaly score 
    depends on how isolated the object is with respect to 
    the surrounding neighborhood
    
    Parameters
    ----------
    X : dataframe object, of shape (n_samples, n_features)
    \t Training data. All elements of X must be finite, 
    \t i.e. no NaNs or infs.
        
    **params : LocalOutlierFactor properties, optional
    \t params are used to specify or override properties 
    \t of sklearn.neighbors.LocalOutlierFactor such as 
    \t n_neighbors, p, contamination.
    \t see https://scikit-learn.org for more info
    
    Returns
    -------
    dictionary of
    - model  : fitted LocalOutlierFactor
    - scores : LOF score array of shape (n_sample,)
    - threshold : Minimum score for outliers
    '''
    params = update_params(LocalOutlierFactor,params)
    data = dict(scores=None, threshold=None, model=None)
    
    # Local-Outlier-Factor
    lof = LocalOutlierFactor(**params)
    lof.fit(X)
    data['model'] = lof
    data['scores'] = abs(lof.negative_outlier_factor_)
    data['threshold'] = abs(lof.offset_)
    return data

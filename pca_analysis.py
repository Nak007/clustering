'''
Class
-----
(1) outliers
(2) principal_components
(3) plot_factors
(4) factor_analysis
'''
import pandas as pd, numpy as np, os, math
import matplotlib.pylab as plt
from matplotlib import cm
from scipy import stats
from scipy.stats import gaussian_kde
from factor_analyzer import FactorAnalyzer
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

class outliers:
  
    '''
    ** Capping Outliers **
    
    Each of approaches fundamentally predetermines lower and upper bounds 
    where any point that lies either below or above those points is 
    identified as outlier. Once identified, such outlier is then capped at 
    a certain value above the upper bound or floored below the lower bound.

    1) Percentile : (alpah, 100-alpha)
    2) Sigma : (average - beta * sigma, average + beta + sigma)  
    3) Interquartile Range (IQR) : (Q1 - beta * IQR, Q3 + beta * IQR)
    4) Gamma : (min(IQR,gamma), max(IQR,gamma))
    \t This approach determines rate of change (r) and locates cut-off 
    \t (different alpha at both ends) where "r" is found to be the highest 
    \t within given range. Nevertheless, this may produce an unusally big 
    \t "alpha" causing more to be outliers, especially when range is 
    \t considerably large. Therefore, outputs are then compared against 
    \t results from "Interquartile Range" as follow;
    
    Parameters
    ----------
    method : str, optional, (default: 'gamma')
    \t Method of cappling outliers i.e. 'percentile', 'interquartile', 
    \t 'sigma', and 'gamma'
    
    pct_alpha : float, optional, (default: 1.0)
    \t Alpha (pct_alpha) refers to the likelihood that the population lies  
    \t outside the confidence interval. Alpha is usually expressed as a 
    \y proportion or in this case percentile e.g. 1 (1%)
    
    beta_sigma : float, optional, (default: 3.0)
    \t Beta (beta_sigma) is a constant that refers to amount of standard 
    \t deviations away from its mean of the population
    
    beta_iqr : float, optional, (default: 1.5)
    \t beta (beta_iqr) is a muliplier of IQR
    
    pct_limit : float, optional, (default: 10)
    \t This only applies when "gamma" is selected. It limits the boundary
    \t of both tails for gammas to be calculated
    
    n_interval : int, optional, (default: 100)
    \t This only applies when "gamma" is selected. It refers to number of 
    \t intervals where gammas are calculated
    
    Methods
    -------
    fit(self, X) : fit model according to training data
    '''
    
    def __init__(self, method='gamma', pct_alpha=1.0, beta_sigma=3.0, beta_iqr=1.5, 
                 pct_limit=10, n_interval=100):

        self.method = method
        self.pct_alpha = pct_alpha
        self.beta_sigma = beta_sigma
        self.beta_iqr = beta_iqr
        
        self.n_interval = n_interval
        # index that devides left and right tails (middle)
        self.mid_index = int(n_interval*0.5)
        self.low_pct = int(n_interval*pct_limit/100)
        
    def __to_array(self, X):
    
        if isinstance(X, (pd.core.series.Series, list)):
            return [X.name], np.array(a).reshape(-1,1)
        elif isinstance(X, pd.core.frame.DataFrame):
            return list(X), X.values
        elif isinstance(X, np.ndarray) & (X.size==len(X)):
            return ['Unnamed'], np.array(X).reshape(-1,1)
        elif isinstance(X, np.ndarray) & (X.size!=len(X)):
            digit = 10**math.ceil(np.log(X.shape[1])/np.log(10))
            columns = ['Unnamed: '+str(digit+n)[1:] for n in range(X.shape[1])]
            return columns, X
    
    def __iqr_cap(self, a):

        q1, q3 = np.nanpercentile(a,25), np.nanpercentile(a,75)
        low = q1 - (q3-q1) * self.beta_iqr
        high = q3 + (q3-q1) * self.beta_iqr
        return low, high
  
    def __sigma_cap(self, a):

        mu, sigma = np.nanmean(a), np.nanstd(a)
        low = mu - sigma * self.beta_sigma
        high = mu + sigma * self.beta_sigma
        return low, high
  
    def __pct_cap(self, a):

        low = np.nanpercentile(a,self.pct_alpha)
        high = np.nanpercentile(a,100-self.pct_alpha)
        return low, high
  
    def __delta_gamma(self, X, delta_asec=True, gamma_asec=True):

        '''
        Determine change (delta), and rate of change (gamma)
        
        Parameters
        ----------
        X : 1D-array
        \t An array, whose members are arrange in monotonically 
        \t increasing manner 
        
        delta_asec : bool, optional, (default: True)
        \t If True, deltas are ascendingly ordered 
        
        gamma_asec : bool, optional, (default: True)
        \t If True, gammas are ascendingly ordered
        '''
        # Slope (1st derivative, delta)
        diff_X = np.diff(X)
        divisor = abs(X.copy()); divisor[divisor==0] = 1
        if delta_asec: delta = diff_X/divisor[:len(diff_X)]
        else: delta = diff_X/-divisor[1:]

        # Change in slope (2nd derivative, gamma)
        diff_del = np.diff(delta)
        divisor = abs(delta); divisor[divisor==0] = 1
        if gamma_asec: gamma = diff_del/divisor[:len(diff_del)]
        else: gamma = diff_del/-divisor[1:]
       
        return delta, gamma
  
    def __gamma_cap(self, X):

        # Create range of percentiles
        p_range = np.arange(self.n_interval+1)/self.n_interval*100
        a = np.array([np.nanpercentile(X,p) for p in p_range])
        r = self.__iqr_cap(X)

        # Low side delta and gamma. Gamma is arranged in reversed order 
        # as change is determined towards the lower number (right to left)
        gamma = self.__delta_gamma(a, gamma_asec=False)[1]
        chg_rate = gamma[:(self.mid_index-1)]
      
        # Low cut-off and index of maximum change (one before)
        min_index = np.argmax(chg_rate[:self.low_pct]) + 1
        low_cut = min_index/self.n_interval*100 # convert to percentile
        low = min(np.percentile(a, low_cut), r[0])

        # Recalculate for high-side delta and gamma (ascending order)
        gamma = self.__delta_gamma(a)[1] 
        chg_rate = gamma[self.mid_index:]
        
        # High cut-off and index of maximum change (one before)
        max_index = np.argmax(chg_rate[-self.low_pct:])-1
        max_index = self.mid_index + max_index - self.low_pct
        high_cut = (max_index/self.n_interval*100)+50 # convert to percentile
        high = max(np.percentile(a, high_cut), r[1])
        
        return low, high
  
    def fit(self, X):
        
        '''
        Parameters
        ----------
        X : array-like, sparse matrix, of shape (n_samples, n_features)
        \t Sample data
        
        Returns
        -------
        self.limit_ : dictionary object, shape of (n_features, 3)
        \t A dictionary of lower and upper limits for respective variables
        
        self.capped_X : dictionary object, shape of (n_samples, n_features)
        \t A dictionary of capped variables 
        '''
        columns, a = self.__to_array(X); c = [None]*len(columns)
        self.limit_ = dict(variable=columns, lower=c.copy(), upper=c.copy())
        
        for n in range(a.shape[1]):
    
            k = a[:,n][~np.isnan(a[:,n])]
            if self.method == 'gamma':
                r = self.__gamma_cap(k)   
            elif self.method == 'interquartile': 
                r = self.__iqr_cap(k)
            elif self.method == 'sigma': 
                r = self.__sigma_cap(k)
            elif self.method == 'percentile': 
                r = self.__pct_cap(k)
            
            # compare against actual data
            low = max(r[0], min(k))
            high = min(r[1], max(k))
            
            # replace data in array
            k[(k>high)], k[(k<low)] = high, low
            a[:,n][~np.isnan(a[:,n])] = k
            self.limit_['lower'][n] = low
            self.limit_['upper'][n] = high
         
        self.capped_X = pd.DataFrame(a,columns=columns).to_dict(orient='list')

class principal_components:
  
    '''
    ** Principal Component Analysis **
    
    The eigenvectors and eigenvalues of a covariance (or correlation) 
    matrix represent the "core" of a PCA: The eigenvectors 
    (principal components) determine the directions of the new feature 
    space, and the eigenvalues determine their magnitude. 
    In other words, the eigenvalues explain the variance of the data 
    along the new feature axes.
    
    In order to decide which eigenvector(s) can be dropped without 
    losing too much information for the construction of 
    lower-dimensional subspace, we need to inspect the corresponding 
    eigenvalues: The eigenvectors with the lowest eigenvalues bear the 
    least information about the distribution of the data; those are the 
    ones to be dropped. In order to do so, the common approach is to 
    rank the eigenvalues from highest to lowest in order choose the top 
    k eigenvectors.
    
    Parameters
    ----------
    var_cutoff : int, optional, (default:80)
    \t Explained Variance threshold
    
    eigen_cutoff : float, optional, (default:1.0)
    \t Eigen value threshold
    
    Methods
    -------
    - fit(self, X) : Fit the model according to the given training data
    - transform(self, X) : Apply dimensionality reduction to X.
    - plot(self[, fname]) : plot cumulative and proportional variances and 
      eigen values given training data
    - explained_variance(self, axis) : plot explained variances for all PCs
    - eigen_value(self, axis) : plot eigen values for all PCs 
    - plot_factor_loadings(self[, n_step, fname]) : plot factor loadings
    - factor_loadings(self, fig, axis[, n_step]) : plot factor loadings
    
    Attributes
    ----------
    self.var_cutoff
    self.eigen_cutoff
    '''
    def __init__(self, var_cutoff=80, eigen_cutoff=1.0):
    
        self.var_cutoff = var_cutoff
        self.eigen_cutoff = eigen_cutoff
  
    def __to_array(self, X):
    
        if isinstance(X, (pd.core.series.Series, list)):
            return [X.name], np.array(a).reshape(-1,1)
        elif isinstance(X, pd.core.frame.DataFrame):
            return list(X), X.values
        elif isinstance(X, np.ndarray) & (X.size==len(X)):
            return ['Unnamed'], np.array(X).reshape(-1,1)
        elif isinstance(X, np.ndarray) & (X.size!=len(X)):
            digit = 10**math.ceil(np.log(X.shape[1])/np.log(10))
            columns = ['Unnamed: '+str(digit+n)[1:] for n in range(X.shape[1])]
            return columns, X
  
    def fit(self, X):
        
        '''
        Parameters
        ----------
        X : array-like, sparse matrix, of shape (n_samples, n_features)
        \t Sample data
        
        Returns
        -------
        self.correl : array of float, shape of (n_features, n_features)
        \t Correlation matrix of all features
        
        self.eig_value : array of float, shape of (n_features,)
        \t The sum of all eigenvalues is the total variance explained. 
        \t The eigenvalue of a factor divided by the sum of the eigenvalues 
        \t is the proportion of variance explained by that factor.
        
        self.eig_vector : array of float, shape of (n_features, n_features)
        \t An eigenvector is a vector whose direction remains unchanged when 
        \t a linear transformation is applied to it
        
        self.loadings : array of float, shape of (n_features, n_features)
        \t The loading matrix shows the relationship between variables 
        \t and new principal components, similar to Pearson correlation
        '''
        # find correlation matrix
        self.columns, a = self.__to_array(X)
        #self.correl = np.corrcoef(X.T)
        self.correl = pd.DataFrame(a).corr().values

        # find eigen vectors and eigen values
        eigvalues, eigvectors = np.linalg.eig(self.correl)
        
        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs = [(eigvalues[n], eigvectors[:,n].reshape(-1,1)) for n in range(X.shape[1])]
        eig_pairs.sort(reverse=True, key=lambda x:x[0]) #<-- ascending order
        eig_pairs = np.array(eig_pairs)
        
        # eigenvalues and eigenvectors
        self.eig_value = eig_pairs[:,0].ravel().astype(float)
        self.eig_vector = np.hstack(eig_pairs[:,1]) #<-- Orthonormal       
        self.var_exp = eig_pairs[:,0]/sum(eigvalues)*100
        self.cum_var_exp = np.cumsum(self.var_exp)

        # Factor Loading
        self.loadings = self.__loadings()
  
    def __loadings(self):
        
        '''
        ** Loadings (scaled by 1-stdev) **
        
        It is a way of weighting the "more important" dimension 
        more highly. For rooting the eigenvalue, recall that the 
        eigenvalues in PCA represent variances, and so the square 
        root of the eigenvalue represents the standard deviation. 
        It is more natural to scale by standard deviation, since 
        it is in the same units as the data  So, by scaling by the 
        rooted eigenvalues, it is like we are looking 1 standard 
        deviation out, in each of the two PCA dimensions.
        
        Note: Orthonormal matrix has properties that each column 
        vector has length one and is orthogonal to all the other 
        colum vectors (independent) ==> QT.Q= 1
        '''
        a = self.eig_vector * np.sqrt(self.eig_value)
        n_comps = len(self.columns)
        digit = 10**math.ceil(np.log(n_comps+1)/np.log(10))
        columns = ['PC_'+str(i+digit)[1:] for i in range(1,n_comps+1)]
        a = pd.DataFrame(a, columns=columns)
        return a.set_index(pd.Series(self.columns))

    def transform(self, X, n_comps=0):

        '''
        ** Principal Component Score **
        Transform Xs to new dimension given factor loadings
        
        X : dataframe object, of shape (n_samples, n_features)
        \t X should be standardized.
        
        n_comps : int, optional, (default:0)
        \t Number of principal components
        '''
        a = X[self.columns].copy().values # <-- order of columns
        a = np.array(a.dot(self.loadings.values[:,:n_comps]))
        return pd.DataFrame(a,columns=list(self.loadings)[:n_comps])
  
    def explained_variance(self, axis):
         
        '''
        Plot explained variance for all principal components
        
        Parameters
        ----------
        axis : matplotlib axis object
        \t axis refers to a single Axes object 
        '''
        colors = ('#ff5252','#3c40c6')
        labels = ['Cumulative variances','Proportional variance']
        
        x, line = np.arange(len(self.var_exp)), [None]*2
        kwargs = dict(lw=1.3, marker='o', markersize=4)
        for (n,y) in enumerate([self.cum_var_exp, self.var_exp]):
            line[n] = axis.plot(x, y, **{**kwargs, **dict(color=colors[n])})
        
        kwargs = dict(color='#3d3d3d', fontsize=11)
        axis.set_xlabel(r'$\mathbf{Principal}$ $\mathbf{Components}$', **kwargs)
        axis.set_ylabel(r'$\mathbf{Explained}$ $\mathbf{Variance}$ (%)', **kwargs)
        axis.set_facecolor('white')
        axis.set_title('PCA: Explained Variance', fontsize=14)
        
        axis.set_xticks(x)
        kwargs = dict(color='#3d3d3d', fontsize=9, rotation=0)
        axis.set_xticklabels(self.__xticklabels(len(x)), **kwargs)
        axis.grid(True, color='#aaa69d', lw=0.5, ls='--')
        axis.set_ylim(-10, 110)
        axis.set_xlim(-0.5,len(x)-0.5)

        # General selection criterion (cum_var) >= 80%)
        n = sum((self.cum_var_exp<=self.var_cutoff).astype(int))-1
        span = axis.axvspan(-0.5, n, color='#aaa69d', alpha=0.2)
        axis.axvline(n, color='#3d3d3d', ls='--', lw=0.8)
        kwargs = dict(fontsize=10, color='#3d3d3d',va='center', ha='right', rotation=90)
        axis.text(n, 50, '# of PCs = %d ' % (n+1), **kwargs)

        lines = [line[0][0], line[1][0], span]
        labels = labels + [r'Cut-off $\leq$ %.2f ' % (self.var_cutoff)]
        axis.legend(lines, labels,loc='best', fontsize=10)

        # Add annotate 
        kwargs = dict(ha='center',va='bottom',fontsize=10)
        for n,amt in enumerate(zip(self.cum_var_exp, self.var_exp)):
            for i in range(2):
                k = {**kwargs,**dict(color=colors[i])}
                axis.annotate(('%.1f' % amt[i]),(n, amt[i]+2), **k)

    def __xticklabels(self, n_components):

        n = n_components + 1
        digit = 10**math.ceil(np.log(n)/np.log(10))
        return ['PC_'+str(i+digit)[1:] for i in range(1,n)]
    
    def eigen_value(self, axis):  

        '''
        Plot explained variance for all principal components
        
        Parameters
        ----------
        axis : matplotlib axis object
        \t axis refers to a single Axes object 
        '''
        x = np.arange(len(self.eig_value))
        line = axis.plot(x, self.eig_value, color='#ff5252', lw=1.3, marker='o', markersize=4)
        kwargs = dict(color='#3d3d3d', fontsize=11)
        axis.set_xlabel(r'$\mathbf{Principal}$ $\mathbf{Components}$', **kwargs)
        axis.set_ylabel(r'$\mathbf{Eigen}$ $\mathbf{Value}$', **kwargs)
        axis.set_facecolor('white')
        axis.set_title('PCA: Scree Plot',fontsize=14)
        axis.set_xticks(x)
        axis.set_xlim(-0.5,len(x)-0.5)
        kwargs = dict(color='#3d3d3d', fontsize=9, rotation=0)
        axis.set_xticklabels(self.__xticklabels(len(x)+1), **kwargs)
        axis.grid(True, color='#aaa69d', lw=0.5, ls='--')

        # General selection criterion (eig_val_np) >= 1)
        n = sum((self.eig_value>=self.eigen_cutoff).astype(int))-1
        span = axis.axvspan(-0.5,n, color='#aaa69d', alpha=0.2)
        axis.axvline(n, color='#3d3d3d', ls='--', lw=0.8)
        kwargs = dict(fontsize=10, color='#3d3d3d', va='top', ha='right', rotation=90)
        e_max = max(self.eig_value)
        axis.text(n, e_max, '# of PCs = %d ' % (n+1), **kwargs)

        lines = [line[0], span]
        labels = ['Eigin value', r'Cut-off $\geq$ %.2f ' % (self.eigen_cutoff)]
        axis.legend(lines, labels,loc='best', fontsize=10)

        # Add annotate
        kwargs = dict(color='#ff5252', ha='center',va='bottom',fontsize=10)
        for n, amt in enumerate(self.eig_value):
            axis.annotate('{:.2f}'.format(amt),(n,amt+0.2), **kwargs)

    def plot(self, fname=None):

        '''
        This function plots Explained Variance, and Scree 
        plot of eigen values
        
        Parameters
        ----------
        fname : str or PathLike or file-like object (default:None)
        \t A path, or a Python file-like object 
        \t (more info see matplotlib.pyplot.savefig)
        '''
        width = len(self.eig_value)*0.6
        fig, axis = plt.subplots(2,1,figsize=(width,8))
        self.explained_variance(axis[0])
        self.eigen_value(axis[1])
        fig.tight_layout()
        if fname!=None: plt.savefig(fname)
        plt.show()

    def plot_factor_loadings(self, n_step=15, fname=None):
        
        '''
        Plot factor loadings
        
        Parameters
        ----------
        n_step : int, optional, (default: 15)
        \t Number of color steps applied to loading matrix
        
        fname : str or PathLike or file-like object (default:None)
        \t A path, or a Python file-like object 
        \t (more info see matplotlib.pyplot.savefig)
        '''
        height = len(self.columns)*0.5
        width = len(self.eig_value)*0.8
        fig, axis = plt.subplots(figsize=(width,height))
        self.factor_loadings(fig, axis, n_step)
        fig.tight_layout()
        if fname!=None: plt.savefig(fname)
        plt.show()
        
    def factor_loadings(self, fig, axis, n_step=15):
        
        '''
        Plot factor loadings
        
        Parameters
        ----------
        fig : matplotlib figure object
        \t The top level container for all the plot elements
        
        axis : matplotlib axis object
        \t axis refers to a single Axes object
        
        n_step : int, optional, (default: 15)
        \t Number of color steps applied to loading matrix
        '''
        kwargs = dict(cmap=cm.get_cmap('RdBu',n_step), edgecolors='#4b4b4b', 
                      linewidths=0.2, vmin=-1, vmax=1)
        c = axis.pcolor(self.loadings, **kwargs)
        
        kwargs = dict(color='#3d3d3d', fontsize=11)
        axis.set_xlabel(r'$\mathbf{Principal}$ $\mathbf{Components}$', **kwargs)
        axis.set_ylabel(r'$\mathbf{Variables}$', **kwargs)
        axis.set_title('PCA: Factor Loadings (Component Matrix)',fontsize=14)
        
        kwargs = dict(color='#3d3d3d', fontsize=9, rotation=0)
        x = np.arange(len(self.eig_value)) + 0.5
        axis.set_xticks(x)
        axis.set_xticklabels(self.__xticklabels(len(x)+1), **kwargs)
        
        axis.set_yticks(np.arange(len(self.columns))+0.5)
        axis.set_yticklabels(self.columns, color='#3d3d3d', fontsize=10)
        fig.colorbar(c, ax=axis)
        
class plot_factors:
  
    '''
    Using kernel density estimation, this function plots
    chosen factor against the rest of remaining factors
    
    Parameters
    ----------
    random_state : int, optional, (defualt:0)
    \t Seed for the random number generator
    
    frac : float, optional, (default:0.5)
    \t Percentage of samples to be randomly selected.
    \t Fraction of axis items to return. 
    
    alpha : float, optional, (default: 1.0)
    \t Alpha refers to the probability outside the confidence 
    \t interval, meaning that any sample that lies outside 
    \t such interval is ignored in the plot
    
    n_column : int, optional, (default: 4)
    \t Number of display columns
    
    Methods
    -------
    fit(self, X[, factor, fname]) : plot factors
    '''
    def __init__(self, random_state=0, frac=0.5, alpha=1, n_column=4):
        
        self.kwargs = dict(random_state=random_state,frac=frac)
        self.alpha = alpha
        self.n_column = n_column
        self.cmap = cm.get_cmap('OrRd',10)

    def __plot(self, fig, axis, x1, x2):
        
        xx = np.vstack([np.array(n).ravel() for n in [x1,x2]])
        kwargs = dict(c=gaussian_kde(xx)(xx), s=10, alpha=0.8, cmap=self.cmap)
        c = axis.scatter(x1, x2, **kwargs)
        kwargs = dict(color='#3d3d3d', fontsize=10, fontweight='bold')
        axis.set_xlabel('%s' % x1.name, **kwargs)
        axis.set_ylabel('%s' % x2.name, **kwargs)
        axis.set_facecolor('white')
        axis.set_xlim(tuple([np.nanpercentile(x1,a) for a in [self.alpha,100-self.alpha]]))
        axis.set_ylim(tuple([np.nanpercentile(x2,a) for a in [self.alpha,100-self.alpha]]))
        #axis.tick_params(axis='both',bottom=False, left=False, labelbottom=False, labelleft=False)
        fig.colorbar(c, ax=axis)
 
    def fit(self, X, factor=None, fname=None):

        '''
        Parameters
        ----------
        X : dataframe object, of shape (n_samples, n_features)
        \t Sample data
        
        factor : str, optional, (default:None)
        \t Name of factor that is used for comparison against
        \t the rest of factors. If not defined, the first one
        \t from dataframe is selected.
        
        fname : str or PathLike or file-like object (default:None)
        \t A path, or a Python file-like object 
        \t (more info see matplotlib.pyplot.savefig)
        '''
        # random sampling
        sample = X.sample(**self.kwargs)

        # plot layout
        n_figure = X.shape[1]-1
        n_row = int(math.ceil(n_figure/self.n_column))
        gridsize = (n_row, self.n_column)
        fig_loc = [(n,m) for n in range(n_row) 
                   for m in range(self.n_column)][:n_figure]
        
        # Columns names
        if factor is None: factor = list(X)[0]
        columns = [n for n in list(X) if n != factor]
        fig = plt.figure(figsize=(self.n_column*4, n_row*3))
        for (n,loc) in enumerate(fig_loc):
            axis = plt.subplot2grid(gridsize, loc)
            self.__plot(fig, axis, sample[factor], sample[columns[n]])
        fig.tight_layout()
        if fname!=None: plt.savefig(fname)
        plt.show()
        
class factor_analysis:
  
    '''
    ** Factor Analysis **
    
    PCA often needs rotation for easier interpretation. 
    Factor rotation tries to maximize variance of the squared 
    loadings in each factor so that each factor has only a few 
    variables with large loadings and many other variables with 
    low loadings. Rotation can proceed by rotating two principal 
    components successively or all principal components 
    simultaneously

    Note : Module details
    https://factor-analyzer.readthedocs.io/en/latest/
    factor_analyzer.html#factor-analyzer-analyze-module
    
    Parameters
    ----------
    loadings : dataframe object, of shape (n_features, n_features)
    \t The loading matrix obtained from PCA represents the 
    \t relationship between variables and principal components, 
    \t similar to Pearson correlation. In addition, loadings must
    \t have variable name as its index.
    
    Methods
    -------
    - fit(self, X[, n_factors, rotation]) : Fit the model according 
      to the given training data
    - transform(self, X) : Apply dimensionality reduction to X
    - plot(self[, fname]) : plot loading matrix
     '''
    def __init__(self, loadings):

        self.loadings = loadings.copy()
        self.features = loadings.index.values

    def fit(self, n_factors=3, rotation='varimax'):
        
        '''
        Parameters
        ----------
        n_factors : int, optional, (default:3)
        \t The number of factors to select
        
        rotation : str, optional, (default:'varimax')
        \t The type of rotation to perform after fitting 
        \t the factor analysis model
        
        \t Rotation Methods
        \t (a) varimax : orthogonal rotation
        \t (b) promax  : oblique rotation
        \t (c) oblimin : oblique rotation
        \t (d) oblimax : orthogonal rotation
        \t (e) quartimin : oblique rotation
        \t (f) quartimax : orthogonal rotation
        \t (g) equamax : orthogonal rotation
        
        Returns
        -------
        self.variance : array of floats
        \t Calculate the factor variance information, 
        \t including variance, proportional variance and 
        \t cumulative variance for each factor

        self.loadings_ : array of floats, 
        of shape(n_factors, n_factors)
        \t The factor loadings matrix
        '''
        self.n_factors, self.rotation = n_factors, rotation
        kwargs = dict(n_factors=n_factors, rotation=rotation, 
                      is_corr_matrix=True)
        fa = FactorAnalyzer(**kwargs)
        fa.fit(self.loadings)
        self.variance = fa.get_factor_variance()
        self.loadings_ = fa.loadings_
    
    def transform(self, X):

        '''
        ** Rotated Factor Score **
        Get the factor scores for new data set
        
        Parameters
        ----------
        X : dataframe object, of shape (n_samples, n_features)
        \t X must be standardized.
        '''
        m = 10**math.ceil(np.log(self.n_factors)/np.log(10))
        columns = ['Factor_' + str(i+m)[1:] 
                   for i in range(1,self.n_factors+1)]
        a = X[self.features].values
        a = np.array(a.dot(self.loadings_))
        return pd.DataFrame(a,columns=columns)
  
    def plot(self, n_step=15, fname=None):

        '''
        Plot rotated component matrix
        
        Parameters
        ----------
        n_step : int, optional, (default: 15)
        \t Number of color steps applied to loading matrix
        
        fname : str or PathLike or file-like object (default:None)
        \t A path, or a Python file-like object 
        \t (more info see matplotlib.pyplot.savefig)
        '''
        figsize = (self.n_factors,len(self.features)*0.5)
        fig, axis = plt.subplots(figsize=figsize)
        ticks = np.arange(self.n_factors) + 0.5
        kwargs = dict(cmap=cm.get_cmap('RdBu',n_step), vmin=-1, vmax=1, 
                      edgecolors='#4b4b4b', linewidths=0.2)
        c = axis.pcolor(self.loadings_, **kwargs)
        title = '\n'.join(('Rotated Component Matrix',
                           'Rotation Method: %s' % self.rotation))
        axis.set_title(title, fontsize=12)
        kwargs = dict(color='#3d3d3d', fontsize=10)
        axis.set_xlabel(r'$\bf{Components}$', **kwargs)
        axis.set_ylabel(r'$\bf{Variables}$', **kwargs)
        axis.set_xticks(ticks)
        xticklabels = ['Factor_' + str(i) for i in range(1,self.n_factors+1)]
        axis.set_xticklabels(xticklabels, color='#3d3d3d', fontsize=9)
        axis.set_yticks(np.arange(len(self.features)) + 0.5)
        axis.set_yticklabels(self.features, color='#3d3d3d', fontsize=10)
        
        fig.colorbar(c, ax=axis)
        fig.tight_layout()
        if fname!=None: plt.savefig(fname)
        plt.show()

class variable_cluster:
    
    '''
    ** Variable Clustering **
    
    The "variable_cluster" algorithm proceeds as the following steps:
    
    (1) Principal Components (PC) are computed by using linear 
        decomposition technique, namely Principal Component Analysis 
        (PCA), of a correlation matrix.
    (2) Select the number of PCs from percentage of explained variance
        (proportion) or minimum eigen value (mineigen).
    (3) By using obtained PCs in absolute term, hierarchical clustering 
        (linkage) is performed in order to split variables into clusters 
        (maxclusters).
    (4) For each variable, the ratio of (1-r(k))/(1-r(n)) is computed,
        where r(k) and r(n) represent average of r-sqaures with other 
        variables within the same cluster, in the nearest cluster, 
        respectively. Small values of this ratio indicate good 
        clustering.
 
    Parameters
    ----------
    maxclusters : int, optional, (default:10)
    \t The number of clusters, in which function computes
    
    proportion : int, optional, (default:80)
    \t The percentage of variance explained that a pool of 
    \t principal components has to collectively reach as one 
    \t of stopping criteria
    
    mineigen : float, optional, (default:1.0)
    \t The minimum eigen value
    
    Methods
    -------
    self.fit : fit model according to given variables
    self.plot_varclus : visualize clustering in dendrogram plot
    self.plot_loadings : loading plot of selected principal components
    '''
    def __init__(self, maxclusters=10, proportion=80, mineigen=1.0):
        
        self.maxclusters = max(maxclusters,2)
        self.proportion = proportion/100
        self.mineigen = mineigen
    
    def fit(self, X):
        
        '''
        Parameters
        ----------
        X : dataframe object, of shape (n_samples, n_features)
        
        Returns
        -------
        - self.n_princom : Number of Principal Components
        - self.loadings : Principal component loading matrix
        - self.Z : The hierarchical clustering encoded as a linkage matrix
        - self.r_square : Dataframe object of clustered variables
        '''
        # Fit prinical_components model
        pca = principal_components(self.proportion*100,self.mineigen)
        pca.fit(X)
        
        # select number of principal components
        n_pc1 = (pca.eig_value>self.mineigen).sum()
        cum_var = np.cumsum(pca.var_exp/sum(pca.var_exp))
        n_pc2 = (cum_var<=self.proportion).sum()
        
        # determine clusters
        self.n_princom = max(n_pc1, n_pc2, 2)
        self.loadings = pca.loadings.values[:,:self.n_princom]       
        self.Z = linkage(abs(self.loadings), method='ward', metric='euclidean')
        labels = np.hstack([fcluster(self.Z, k, criterion='maxclust').reshape(-1,1) 
                            for k in np.arange(1,self.maxclusters+1)])
        
        # This is neccessary to reduce matrix when algorithm can no longer
        # split variables into sub cluster and identical labels are produced
        # instead.
        self.maxclusters = sum([~((labels[:,n-1]==labels[:,n]).sum()==labels.shape[0]) 
                                for n in np.arange(1,labels.shape[1])]) + 1
        self.labels = labels[:,:self.maxclusters]
        
        # find own-cluster and nearest-cluster R-square
        a = self.__own_cluster_r(X, self.labels)
        b = self.__nearest_cluster_r(X, self.labels)
        r = a.merge(b.drop(columns=['cluster']),on=['variable'],how='left')
        r['r_ratio'] = (1-r['own_cluster'])/(1-r['nearest_cluster'])
        r = r.sort_values(by=['cluster','r_ratio'],ascending=[True,True])
        self.r_square = r.reset_index(drop=True)
        
    def __own_cluster_r(self, X, labels):

        '''
        Determine R-squared of variables within the cluster
        
        Parameters
        ----------
        X : dataframe object, of shape (n_samples, n_features)
        labels : Array of labels, of shape (n_features, n_clusters)
        '''
        clusters = labels[:,-1]
        a = [(n,f) for n,f in zip(clusters,list(X))]
        a.sort(key=lambda a: a[0])
        sort_keys = np.array(a)[:,1].ravel()
        corr = X[sort_keys].corr()
        
        # Construct matrix for own cluster
        a = np.sort(clusters).reshape(-1,1)
        c = np.sqrt(a*a.T)
        c = np.where(c==a,corr.values,np.nan)
        
        # eliminate diagonal elements
        c = np.where(np.identity(len(c))==1,np.nan,c)
        # group that has one member is replace with 1
        c = np.where((~np.isnan(c)).sum(axis=1)==0,1,c**2)
        r = np.nanmean(c,axis=1).ravel()
        data = dict(variable=sort_keys.ravel(), cluster=a.ravel(), own_cluster=r)
        return pd.DataFrame(data)
    
    def __pairwise_nearest_k(self, labels):
  
        '''
        Determine the closest cluster for all clusters

        Parameters
        ----------
        labels : Array of integers, of shape (n_features,n_clusters)
        '''
        # Array of labels
        p = np.unique(labels,axis=0)
        z = np.zeros((1,labels.shape[1]))
        p = np.vstack((z,p,z))
        # Initial vaues
        cons, c = 2, 1 # cons : consecutiveness, c : nth layer from bottom
        combi = dict((str(n),[]) for n in np.arange(cons,labels.max()))

        while cons<len(p)-2:
            c += 1
            # Contruct an array of consecutiveness
            a = np.hstack([p[:,-c][n:n+len(p)-cons-1].reshape(-1,1) for n in range(2+cons)])
            # Condition (1) : elements that stay in the middle must be the same
            cond1 = (np.diff(a[:,1:cons+1]).sum(axis=1)==0)
            # Condition (2) : both ends of elements must be different
            cond2 = ((a[:,0]!=a[:,1]) & (a[:,-2]!=a[:,-1]))
            # Both conditions must be satisfied in order to be in the same cluster
            a = (cond1 & cond2).astype(int)
            n = [np.arange(n,n+cons).tolist() for n,p in enumerate(a,1) if p==1]
            # if new entry has more member, then accept, otherwise move to
            # next consecutiveness (plus 1)
            if len(combi[str(cons)]) < len(n): combi[str(cons)] = n
            else: cons+=1; c = c - 1

        combi = [e for key in combi.keys() for e in combi[key]]
        # Combination that has max number of elements
        n_max = np.max([len(n) for n in combi])
        z = np.full((len(combi),n_max),0)
        for r,p in enumerate(combi):
            for c,s in enumerate(p):
                z[r,c] = s

        # find the nearest clusters
        nearest_k = dict()
        for k in np.unique(labels):
            n = np.argmax((z==k).sum(axis=1))
            a = np.unique(z[n,:])
            a = np.where(a==k,0,a)
            nearest_k[k] = a[a>0].tolist()
        return nearest_k
    
    def __nearest_cluster_r(self, X, labels):

        '''
        Determine R-squared of variables from the nearest cluster
        
        Parameters
        ----------
        X : dataframe object, of shape (n_samples, n_features)
        labels : Array of labels, of shape (n_features, n_clusters)
        '''
        nearest = self.__pairwise_nearest_k(labels)
        clusters = labels[:,-1]
        a = [(n,f) for n,f in zip(clusters, list(X))]
        a.sort(key=lambda a: a[0])
        sort_keys = np.array(a)[:,1].ravel()
        corr = X[sort_keys].corr().values
        k_list = np.array(a)[:,0].ravel()

        m = np.full(corr.shape,0)
        for k in np.unique(clusters):
            h = np.full(len(k_list),0)
            for n in nearest[k]:
                h = h + (k_list==str(n))
            m += (k_list==str(k)).reshape(-1,1)*h
        r = np.nanmean(np.where(m==1,corr,np.nan)**2,axis=1)
        data = dict(variable=sort_keys.ravel(), cluster=k_list, nearest_cluster=r)
        return pd.DataFrame(data)
    
    def plot_varclus(self, axis):
                   
        '''
        Plot dendrogram of variable clustering
        
        Parameters
        ----------
        axis : matplotlib axis object
        \t axis refers to a single Axes object 
        '''
        kwargs = dict(p=int(self.maxclusters), truncate_mode='lastp', leaf_rotation=90, 
                      leaf_font_size=10, show_contracted=True, ax=axis,orientation='right')
        dendro = dendrogram(self.Z, **kwargs)
        k,n = np.unique(self.labels[:,-1], return_counts=True)
        
        yticklabels = ['C{0} ({1})'.format(s[0],s[1]) for s in zip(k,n)]
        axis.set_yticklabels(yticklabels, rotation=0)
        axis.set_facecolor('white')
        kwargs = dict(color='#3d3d3d', fontsize=11, fontweight='bold')
        axis.set_ylabel('Cluster', **kwargs)
        axis.set_xlabel('Distance', **kwargs)
        axis.set_title('Variable Cluster', fontsize=14)
        
        # annotation for distance
        x = np.array(dendro['dcoord'])
        y = np.array(dendro['icoord'])
        d = dendro['leaves']
        for n in range(x.shape[0]):
            nx, ny = x[n,1], y[n,1:3].mean()
            axis.text(nx,ny,'{0} '.format(d[n]), va='center',ha='right')
    
    def plot_loadings(self, fig, n_column=3, show_labels=False):
        
        '''
        ** Loading plot **

        The loading plot illustrates the coefficients of each variable 
        for the first component versus the coefficients for the rest of 
        selected components.

        Regarding the interpretation, loading plot helps identify which 
        variables have the largest effect on each component. 
        Loadings can range from -1 to 1. Loadings close to -1 or 1 indicate 
        that the variable strongly influences the component. 
        
        Parameters
        ----------
        fig : matplotlib figure object
        \t The top level container for all the plot elements.
        
        n_column : int, optional, (default:3)
        \t Number of display columns
        
        show_labels : bool, optional, (default:False)
        \t If True, labels indicating nth feature in the same
        \t order as X, are shown at the end of vectors
        
        Returns
        -------
        list of axes
        '''
        bbox = dict(boxstyle="circle", fc='w', ec='b', pad=0.25)
        props = dict(color='b', va='center', ha='center', fontsize=9, bbox=bbox)
        PC = self.loadings[:,:self.n_princom]
        
        n_axes = self.n_princom - 1
        n_row = int(np.ceil(n_axes/n_column))
        offset = 0.1 # (% offset from bottom and left)
        w, h = (1-offset)/n_column, (1-offset)/n_row
        # (left, bottom, width, height)
        rect = [(c*w+offset,r*h+offset,0.78*w,0.78*h) for r in range(n_row,-1,-1) 
                for c in range(n_column)][:n_axes]
        axes = [fig.add_axes(r) for r in rect]
        
        for n,axis in enumerate(axes,1):
            c = PC[:,[0,n]]
            for r in range(len(c)):
                x,y = [0,c[r,0]], [0,c[r,1]]
                axis.plot(x, y,color='b', lw=0.5, marker='o', ms=3)
                if show_labels: axis.text(x[1], y[1], '{0}'.format(r+1), **props)
            axis.axhline(0, color='grey', lw=1, ls='--')
            axis.axvline(0, color='grey', lw=1, ls='--')
            axis.set_xlabel('PC_1',fontsize=10)
            axis.set_ylabel('PC_{0}'.format(n+1),fontsize=10)
        return axes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_2D(data, labels, ax, pca_t = True, n_components=2, c1=0, c2=1, title=None):
    """
    Function that plots the data using the 2 first PCA components if the card(feature space) > 2
    TODO: conditions sur c1, c2 < n_components si PCA_T ; sinon < n_features 
    
    Args:
        data (numpy array): Dataset
        TODO faire ça aussi :)

    Returns:
        matplotlib fig: figure de matplotlib
    """
    # work on the copy of dataset
    pts = data.copy()

    # if dim(data) > 2 then reduce using PCA
    if pca_t and data.shape[1] > 2:
        pca = PCA(n_components=n_components)
        pca.fit(pts)
        pts = pca.transform(pts)
        print(f"EVR {pca.explained_variance_ratio_}")
        ax.set_xlabel("PCA: C"+str(c1)) 
        ax.set_ylabel("PCA: C"+str(c2))
    else:
        ax.set_xlabel("Feature "+str(c1)) 
        ax.set_ylabel("Feature "+str(c2))

    # plot each cluster points  with a unique color (using masks)
    for l in set(labels):
        mask = labels == l        
        ax.scatter(pts[mask][:,c1], pts[mask][:,c2], label="c"+str(l))
        if title != None:
            ax.set_title(title)    
    ax.legend()
    return ax

def plot_boundary(clf, X, ax, y=None, h=0.002, title=""):
    h = h ## grid mesh size
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1 
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) ## prediction value by zone
    Z = Z.reshape(xx.shape)
    
    # plt.figure(figsize=[5,5]) ## equal x and y lengths for a squared figure
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)


    ax.scatter(X, X, c=y, s = 100)
    ax.set_title('score : ' + title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()
#     plt.xlim([0,1])
#     plt.ylim([0,1])


def get_statistics(df, col="ari"):
    a = np.array([np.array(i)  for i in df.loc[col]])
    gt_mean = a.mean(axis=0)
    gt_std  = a.std(axis=0)
    return gt_mean, gt_std
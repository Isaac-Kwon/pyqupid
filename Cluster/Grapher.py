import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

"""
Module: Grapher
"""
def PrintTFGrid(G):
    assert isinstance(G, np.ndarray)
    assert G.ndim == 2
    for i in range(G.shape[1]):
        for j in range(G.shape[0]):
            if G[i,j]:
                print("T", end="")
            else:
                print("F", end="")
        print()

def PlotSignalAnalogue(X,Y,Z, ax=None):
    if ax == None:
        ax = plt.axis()
    surf = ax.contourf(X,Y,Z,20, cmap='jet')
    cbar = plt.colorbar(surf, ax=ax)

def PlotPixels(Z, ax=None, wholeplot=False, shownumber=False, numberregion=(1, 9999),fontsize=12, highnumber=False, threshold=None):
    if ax == None:
        ax = plt.axis()
    x = np.arange(0, Z.shape[1], 1)
    y = np.arange(0, Z.shape[0], 1)
    X, Y = np.meshgrid(x,y)
    
    if not Z.dtype == bool:
        Z_ = Z> Z.max()*0.01
        cmap='gist_heat_r'
    else:
        Z_ = Z
        cmap='brg'
    
    if Z_[np.where(Z_)].size != 0:
        min_x, max_x = min(X[Z_])-1, max(X[Z_])+1
        min_y, max_y = min(Y[Z_])-1, max(Y[Z_])+1
    else:
        min_x, max_x, min_y, max_y = 0, 1, 0, 1
    
    if threshold:
        img = ax.imshow(Z>threshold, cmap=cmap)
    else:
        img = ax.imshow(Z, cmap=cmap)

    # Major ticks
    ax.set_xticks(np.arange(0, Z.shape[1], 1))
    ax.set_yticks(np.arange(0, Z.shape[0], 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, Z.shape[1]+1, 1))
    ax.set_yticklabels(np.arange(1, Z.shape[0]+1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, Z.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, Z.shape[0], 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1.5)

    if not wholeplot:
        ax.set_xlim(min_x - 0.2*(max_x-min_x), max_x + 0.2*(max_x-min_x))
        ax.set_ylim(min_y - 0.2*(max_y-min_y), max_y + 0.2*(max_y-min_y))
    if shownumber:
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        for (j,i),label in np.ndenumerate(Z):
            if i<xl[0] or i>xl[1] or j<yl[0] or j>yl[1]:
                continue
            if label>=numberregion[0] and label<=numberregion[1]:
                ax.text(i,j,"%.0f"%(label),ha='center',va='center',fontsize=fontsize)
            # if highnumber:
            #     ax.text(i,j,"%.0f"%(label),ha='center',va='center',fontsize=fontsize)
    
    cbar = plt.colorbar(img, ax=ax)

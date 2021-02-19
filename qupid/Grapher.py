import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def PrintTFGrid(G, truetext = 'T', falsetext = 'F'):
    """
    ```True``` 또는 ```False``` 로 이루어진 행렬 값을 받아 쉘에 정사각형으로 출력하는 함수

    Parameters
    ----------
    G : numpy.ndarray 
        출력할 ```True``` 와 ```False``` 로 이루어진 **2차원** 행렬 
    truetext : str 
        ```True``` 일 경우 출력할 문자 (1글자)
    falsetext : str 
        ```False``` 일 경우 출력할 문자 (1글자)

    Returns 
    -------
    None
    """
    assert isinstance(G, np.ndarray), "Input matrix should be numpy.ndarray"
    assert G.ndim == 2, "Dimension of input matrix should be 2-dimensional matrix"
    assert len(truetext)==1, "Text for True should be single charactor"
    assert len(falsetext)==1, "Text for False should be single charactor"
    for i in range(G.shape[1]):
        for j in range(G.shape[0]):
            if G[i,j]:
                print(truetext, end="")
            else:
                print(falsetext, end="")
        print()

def PlotSignalAnalogue(X,Y,Z, ax=None):
    """
    np.meshgrid 로 메시화된 행렬공간값으로 2D 공간그래프를 그리는 함수

    Parameters
    ----------
    X  : numpy.ndarray 
        np.meshgrid 로 형성된 X 값
    Y  : numpy.ndarray 
        np.meshgrid 로 형성된 Y 값
    Z  : numpy.ndarray 
        계산된 높이값 (Z)
    ax : matplotlib.pyplot.axes

    Returns 
    -------
    None
    """
    if ax == None:
        ax = plt.axis()
    surf = ax.contourf(X,Y,Z,20, cmap='jet')
    cbar = plt.colorbar(surf, ax=ax)

def PlotPixels(Z, ax=None, wholeplot=False, shownumber=False, numberregion=(1, 9999),fontsize=12, highnumber=False, threshold=None):
    """
    ```True``` 또는 ```False``` 로 이루어진 행렬 값을 받아 ```matplotlib.pyplot.imshow``` 함수를 활용하여 True 원소를 표시하거나, 숫자로 이루어진 행렬값을 받아 각 원소별로 가지는 값의 높고 낮음을 ```matplotlib.pyplot.imshow``` 를 활용하여 그리는 함수

    Parameters
    ----------
    Z  : numpy.ndarray 
        그래프에 표기할 행렬
    ax : matplotlib.pyplot.axes
        그래프를 그릴 축
    wholeplot : bool, default=False
        - True : 전체 행렬을 볼 수 있도록 함. 
        - False : 값이 0 초과 또는 True 인 행렬값을 주로 볼 수 있도록 축을 확대함.
    shownumber : bool
        - True : 그래프 위에 숫자를 표기함.
        - False : 그래프 위에 숫자를 표기하지 않음.
    numberregion : tuple of num or list of num, default:(1,9999)
        ```shownumber``` 가 ```True``` 일 경우 표시될 숫자의 크기 범위를 지정
    fontsize : int
        ```shownumber``` 가 ```True``` 일 경우 표시될 숫자의 글꼴 크기를 지정
    highnumber : bool
        (depreciated)
    threshold : int or None
        int  : ```threshold``` 보다 높은 크기의 원소를 그림.
        None : 그래프의 높낮이 (색깔) 이 그 위치에서의 값의 크기를 뜻함.

    Returns 
    -------
    None
    """
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

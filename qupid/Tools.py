import numpy as np
from .Grapher import PrintTFGrid


def ShrinkTFArray(array):
    """
    임의의 ndarray 를 받아 False 만으로 이루어진 외곽의 행 또는 열을 제거하여 압축할 수 있는 함수.

    Parameters
    ----------
    array : numpy.ndarray 
        출력할 ```True``` 와 ```False``` 로 이루어진 **2차원** 행렬 

    Returns 
    -------
    farray : numpy.ndarray
    """
    farray = array
    nzX, nzY = array.nonzero()
    farray = array[nzX.min():(nzX.max()+1), nzY.min():(nzY.max()+1)]
    # print("[%d, %d], [%d, %d]" %(x1, x2, y1, y2))
    return farray
    
if __name__ == "__main__":
    a = [
        [False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False],
        [False, False, False,  True,  True,  True, False, False, False],
        [False, False, False, False,  True,  True, False, False, False],
        [False, False, False, False,  True, False, False, False, False],
        [False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False]
    ]
    array = np.array(a)
    farray = ShrinkTFArray(array)
    PrintTFGrid(array, truetext='O', falsetext='.')
    print("=========================")
    PrintTFGrid(farray, truetext='O', falsetext='.')
    # print(array.nonzero())
    
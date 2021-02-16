# -*- coding: utf-8 -*-
"""
Module: ClusterGenerator
"""

import numpy as np

from pprint import pprint
from scipy.integrate import dblquad

"""Useful Snippets"""

def RMatrix(theta):
    """
        수학적인 정의에 필요.  
        회전행렬을 생성하는 함수
    """
    # theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def PinMatrix(matrix, right=True, lower=True):
    """
        행렬을 상/하/좌/우 한칸씩 밀고 당기는 함수.
    """
    rl = [matrix, np.zeros((matrix.shape[0],1))==1]
    if not right:
        rl.reverse()
    matrix_temp = np.hstack(rl)
    ud = [matrix_temp, np.zeros((1, matrix_temp.shape[1]))==1]
    if not lower:
        ud.reverse()
    matrix_final = np.vstack(ud)
    return matrix_final

def PinMatrixTF(matrix, right=True, lower=True):
    """
        행렬을 상/하/좌/우 한칸씩 밀고 당긴 후 True 값으로 필터링하는 함수
    """
    return PinMatrix(matrix, right, lower) == True

"""# 시그널 클래스 정의
시그널 클래스 (Signal) 가 정의된 이후에 (아날로그 시그널임), ``Signal`` 클래스를 상속한 클래스만이 이후에 시그널로 쓰일 수 있음. (이외에는 전부 AssertionError 로 Assert)
"""

class Signal:
    """
    시그널을 정의하기위한 기반 클래스

    Attributes
    ----------
    None
    """
    def GetPDF(self, X,Y):
        """
        시그널의 Probability Density Distribution 을 생성함.  
        본 클래스는 기반 클래스이므로, 0 값을 반환함
        
        Parameters
        ----------
        X : int or float or numpy.ndarray
        Y : int or float or numpy.ndarray

        Returns
        -------
        0 or numpy.zeros(X.shape)
        """
        if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
            if X.size == Y.size:
                print("The signal is not defined. (this is pure class) Returning zero signal for all")
                return np.zeros(X.shape)
            return 0
    def GetSignal(self, X, Y, amplitude=100):
        """
        특정 위치 / 위치 묶음에서 신호값을 계산함.  
        본 클래스는 기반 클래스이므로, 0 값을 반환함
        
        Parameters
        ----------
        X : int or float or numpy.ndarray
        Y : int or float or numpy.ndarray

        Returns
        -------
        numpy.zeros(X.shape)
        """
        return np.zeros(X.shape)

class GaussianSignal(Signal):
    """
    2D 가우시안 분포를 기준으로 한 시그널

    S 또는 (sx,sy,sdeg) 또는 (sx,sy,srad) 로 정의되어야 한다.

    만약 모든 값이 지정되어 들어갈 경우, 다음과 같은 우선순위로 들어간다.

    1. ``S`` 가 있을 경우, ``S``
    2. ``sdeg`` 가 있을 경우, ``(sx,sy,sdeg)``
    3. ``srad`` 가 있을 경우, ``(sx,sy,srad)``

    Parameters
    ----------
    x0   : int or float
           중심 위치의 x값
    y0   : int or float
           중심 위치의 y값
    amplitude : int or float
           기준가우시안분포에서의 신호배수값 (전체 적분값)
    S    : numpy.ndarray :: shape(2,2) dtype=int or float , optional
           2D 가우시안 함수의 분산행렬
    sx   : int or float , optional
           2D 가우시안 함수의 X 방향분산
    sy   : int or float , optional
           2D 가우시안 함수의 Y 방향분산
    sdeg : int or float , optional
           2D 가우시안 함수의 X,Y방향 분산이 정의되었을 때의 X 축을 기준으로 한 회전각 (Degree)
    srad : int or float , optional
           2D 가우시안 함수의 X,Y방향 분산이 정의되었을 때의 X 축을 기준으로 한 회전각 (Radian)

    Attributes
    ----------
    x0 : int or float
        중심 위치의 x값
    y0 : int or float
        중심 위치의 y값
    amplitude : int or float
        기준가우시안분포에서의 신호배수값 (전체 적분값)
    S : numpy.ndarray :: shape=(2,2) det=1
        분포의 분산행렬
    """
    def __init__(self, x0=0, y0=0, amplitude=100, S=None, sx=1, sy=1, sdeg=None, srad=None):
        self.x0 = x0
        self.y0 = y0
        self.amplitude = amplitude
        if S!=None:
            self.S = S
        elif sx!=None and sy!=None:
            if srad == None and sdeg == None:
                self.S = self.GetDispersion(sx, sy, 0)
            elif srad == None and sdeg != None:
                self.S = self.GetDispersion(sx, sy, np.radian(sdeg))
            elif srad != None:
                self.S = self.GetDispersion(sx, sy, srad)
        else:
            assert False
    def GetDispersion(self, sx, sy, srad):
        """
        *(static method)* X 방향, Y 방향 분산과 분산회전각(Radian) 으로 분산행렬을 구함.

        Parameters
        ----------
        sx : x 축방향 분산
        sy : y 축방향 분산
        srad : x 축을 기준으로 한 분산회전각

        Returns
        ----------
        S : numpy.ndarray :: shape=(2,2), det=1, dtype=float
            분산행렬
        """
        return np.matmul(RMatrix(srad), np.array([[sx,0],[0,sy]]))
    def GetStaticPDF(self, X,Y, x0, y0, S):
        """
        *(static method)* 수학적 정의로부터 값을 찾아냄.

        Parameters
        ----------
        X  : int or float or numpy.ndarray
             구할 위치의 x값
        Y  : int or float or numpy.ndarray
             구할 위치의 y값
        x0 : int or float
             가우시안 함수의 중심값
        y0 : int or float
             가우시안 함수의 중심값
        S  : numpy.ndarray :: shape=(2,2)
             분산행렬

        Returns
        -------
        Z : numpy.ndarray :: shape=X.shape
            X 와 Y 위치에서의 크기값 행렬
        """
        X1 = X-x0
        Y1 = Y-y0
        Si = np.linalg.inv(S)
        X_ = Si[0][0] * X1 + Si[0][1] * Y1
        Y_ = Si[1][0] * X1 + Si[1][1] * Y1
        return np.exp(-0.5*(X_**2+Y_**2))/(2*np.pi*np.linalg.det(S))
    def GetPDF(self, x, y):
        """
        수학적 정의 ``(StaticPDF)`` 에 오브젝트의 중앙위치값인 ``(x0, y0)``, 분산행렬인 ``S`` 를 대입한 가우시안 신호의 실제 확률분포를 구함.

        Parameters
        ----------
        x: int or float or np.ndarray
        y: int or float or np.ndarray (assert y.shape==x.shape)

        Returns
        -------
        result: numpy.ndarray :: shape=x.shape (if int or float (1,))
            ``x, y`` 위치에서의 확률값
        """
        return self.GetStaticPDF(x, y, self.x0, self.y0, self.S)
    def GetSignal(self, x, y):
        return self.amplitude * self.GetPDF(x,y)

"""# Digitizer 정의
Digitizer : 시그널을 검출기의 신호로 변환하는 것.

## 초기화

처음 정의할 때 필요한 것
다음 둘 중 하나만 하면 됨. (둘 다 하면, 1번이 인식됨. 필요시 변수 이름을 특정하여 적당한 ``__init__`` 의 변수만 지정 할 것.)
1. 최소, 최대 x,y 값과 양자화수
    * x최소값, x최대값, x칸수, y최소값, y최대값, y칸수
2. meshgrid 를 통해 생성한 X 와 Y 위치 

## 작동요소

1. ``InjectSignal(signal)`` 변수가 모두 들어가서 적절히 정의가 된 signal 을 주입하면, Digitizer 내의 signals 변수에 리스팅된다.
2. ``GetAnalogSignal(X,Y)`` (``X,Y`` 는 meshgrid 를 통해 형성된 공간좌표값) 을 하면, ``X,Y`` 로 주어진 공간에서의 아날로그 신호를 송출한다.
3. ``GetDigitalSignal_Old()`` 아날로그 시그널을 기준으로, ``Digitizer`` 에 정의된 디지털 그리드에 따라 해당 구역 내에 정의된 시그널 크기를 낸다. (그리드 센터의 값을 구한 다음에 면적을 곱하는 방식)
    * ``GetDigitalSignal()`` 그리드 내부를 전부 적분하는 방식  
      신호가 매우 날카로워서 그리드 센터에 신호가 없을 때의 문제를 해결하기 위함. 그러나 *느림*.
4. ``GetDigitizedSignal()`` 디지털 시그널을 기준으로, ``Digitizer`` 의 디지털 그리드에, 각각의 시그널이 ``on`` 인지 ``off``인지 판단한다
"""

class Digitizer:
    """
    신호를 받은 것을 지정된 그리드에 따라 크기로 변환하고, 점화/비점화상태를 구별하여 출력하는 작업을 하는 클래스

    ``(xmin,xmax,xn,ymin,ymax,yn)`` 와 ``(X,Y)`` 둘 중 하나로 정의되어야 함.

    우선순위는 다음과 같음.

    1. ``(xmin,xmax,xn,ymin,ymax,yn)``
    2. ``(X,Y)``

    Parameters
    ----------
    xmin : int or float
            x축 방향 좌표값의 최소값
    xmax : int or float
            x축 방향 좌표값의 최대값
    xn   : int
            x축 방향으로 분할되는 그리드의 조각 수
    ymin : int or float
            y축 방향 좌표값의 최소값
    ymax : int or float
            y축 방향 좌표값의 최대값
    yn   : int
            y축 방향으로 분할되는 그리드의 조각 수
    X    : numpy.ndarray
            numpy.meshgrid 로부터 생성된 X 그리드
    Y    : numpy.ndarray (assert X.shape==Y.shape)
            numpy.meshgrid 로부터 생성된 Y 그리드

    Attributes
    ----------
    self.threshold : int or float
                     디지털 시그널에서 디지타이즈 시그널로 변환할 때의 기준역치값.
    self.signals   : list of Signal (and its subclass)
                     원본 시그널을 저장하기위한 집합. 리스트의 모든 원소가 ``Signal`` 클래스임.
    self.X         : numpy.ndarray
                     검출기 픽셀 의 X 경계값
    self.Y         : numpy.ndarray (Y.shape == X.shape)
                     검출기 픽셀 의 Y 경계값
    self.centerX   : numpy.ndarray (centerX.shape == X.shape - (1,1))
                     검출기 픽셀 의 X 중심값
    self.centerY   : numpy.ndarray (centerY.shape == centerX.shape)
                     검출기 픽셀 의 Y 중심값
    """
    def __init__(self, xmin=None, xmax=None, xn=None, ymin=None, ymax=None, yn=None, # Spanning with x size, y size, number of their bins
                 X=None, Y=None,  # Spanning with X,Y matrix (meshgrid)
                 threshold=25 
                 ):
        self.threshold = threshold
        self.signals = list()
        if (xmin!=None and xmax!=None and xn!=None and ymin!=None and ymax!=None and yn!=None) :
            self.__SpanningBin(xmin, xmax, xn, ymin, ymax, yn)
            return
        if (X!=None and Y!=None):
            self.__SpanningMeshgrid(X,Y)
            return
        assert False, "GridRange/nbin or Grid with meshed should be inputted."
    def __SpanningBin(self, xmin, xmax, xn, ymin, ymax, yn):
        """
        Parameters
        ----------
        xmin : int or float
               x축 방향 좌표값의 최소값
        xmax : int or float
               x축 방향 좌표값의 최대값
        xn   : int
               x축 방향으로 분할되는 그리드의 조각 수
        ymin : int or float
               y축 방향 좌표값의 최소값
        ymax : int or float
               y축 방향 좌표값의 최대값
        yn   : int
               y축 방향으로 분할되는 그리드의 조각 수

        returns
        -------
        None
        """
        X, Y = np.meshgrid(np.linspace(xmin, xmax, xn+1), np.linspace(ymin, ymax, yn+1))
        self.__SpanningMeshgrid(X,Y)
    def __SpanningMeshgrid(self, X,Y):
        """
        Parameters
        ----------
        X    : numpy.ndarray
                numpy.meshgrid 로부터 생성된 X 그리드
        Y    : numpy.ndarray (assert X.shape==Y.shape)
                numpy.meshgrid 로부터 생성된 Y 그리드
        
        returns
        -------
        None
        """
        assert X.shape == Y.shape
        self.X = X
        self.Y = Y
        self.centerX = 0.5*(X[:,:-1]+X[:,1:])[:-1,:]
        self.centerY = 0.5*(Y[:-1,:]+Y[1:,:])[:,:-1]
    def InjectSignal(self, signal):
        """
        시그널을 주입하는 메소드.
        
        Parameters
        ----------
        signal : Signal or its subclass
        
        Returns
        -------
        None

        Raises
        ------
        AssertionError
            주입한 파라메터가 Signal 이나 Signal 의 서브클래스가 아님.

        See Also
        --------
        .Signal         : for Injecting Signal (but it is null, pure-class)
        .GaussianSignal : Example (not-null) signal
        """
        assert isinstance(signal, Signal)
        # self.AnalogSignal = self.AnalogSignal + signal.GetSignal(self.X,self.Y)
        self.signals.append(signal)
    def ClearSignal(self):
        """
        ``InjectSignal`` 을 통해 주입된 모든 신호를 제거하여 초기화함.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.signals.clear()
    def GetAnalogSignal(self, X, Y):
        """
        ``InjectSignal`` 을 통해 주입된 신호로 주어진 X 와 Y 값 좌표에서의 신호값을 구함.

        Parameters
        ----------
        X : int or float or numpy.ndarray
            구하고자 하는 위치의 X 좌표
        Y : int or float or numpy.ndarray
            구하고자 하는 위치의 Y 좌표

        Returns
        -------
        AnalogSignal : 구하고자 하는 위치(단일, ndarray)에서의 시그널 크기
        
        Raises
        ------
        AssertionError
            파라메터의 위치정보인 ``X`` 와 ``Y`` 가 ``numpy.ndarray`` 일 때, shape 가 동일하지 않음.
        """
        if isinstance(X,np.ndarray) or isinstance(Y,np.ndarray):
            assert X.shape == Y.shape
            AnalogSignal = np.zeros(X.shape)
        else:
            AnalogSignal = 0
        for signal in self.signals:
            AnalogSignal += signal.GetSignal(X,Y)
        return AnalogSignal
    def GetDigitalSignal_Old(self):
        centerSignal_ = self.GetAnalogSignal(self.centerX, self.centerY)
        centerSignal  = centerSignal_ * (PinMatrix(self.X) - PinMatrix(self.X, False, True))[1:-1,1:-1] * (PinMatrix(self.Y) - PinMatrix(self.Y, True, False))[1:-1,1:-1]
        return centerSignal
    def GetDigitalSignal_List(self):
        OnPixels      = self.GetDigitalSignal_Old()>0.1
        x    = self.centerX[OnPixels]
        y    = self.centerY[OnPixels]
        xmin = self.X[PinMatrixTF(OnPixels, True, True)]
        xmax = self.X[PinMatrixTF(OnPixels, False, True)]
        ymin = self.Y[PinMatrixTF(OnPixels, True, True)]
        ymax = self.Y[PinMatrixTF(OnPixels, True, False)]
        X, Y = np.meshgrid(np.arange(0, self.centerX.shape[1], 1), np.arange(0, self.centerX.shape[0], 1))
        X, Y = (X[OnPixels], Y[OnPixels])
        assert xmin.size == x.size
        assert xmin.size == y.size
        assert xmin.size == xmax.size
        assert xmin.size == ymin.size
        assert xmin.size == ymax.size
        ansX = list()
        ansY = list()
        ansZ = list()
        for i in range(xmin.size):
            ansZ.append(dblquad(lambda x,y : self.GetAnalogSignal(x,y), ymin[i], ymax[i], xmin[i], xmax[i])[0]) 
            ansX.append(X[i])
            ansY.append(Y[i])
        return np.array(ansX), np.array(ansY), np.array(ansZ)
    def GetDigitalSignal(self):
        iX,iY,iZ = self.GetDigitalSignal_List()
        Z = np.zeros(self.centerX.shape)
        for i in range(iX.size):
            Z[iY[i], iX[i]] = iZ[i]
        return Z
    def GetDigitizedSignal_Old(self):
        return self.GetDigitalSignal_Old()>self.threshold
    def GetDigitizedSignal_List(self):
        X,Y,Z = self.GetDigitalSignal_List()
        return X[Z>self.threshold], Y[Z>self.threshold]
    def GetDigitizedSignal(self):
        return self.GetDigitalSignal()>self.threshold
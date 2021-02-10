# -*- coding: utf-8 -*-
"""
Module: ClusterGenerator
"""


"""
# Cluster Generator
이중 입자 클러스터를 찾아내기 위한 클러스터 시뮬레이터

필요한 패키지 내장
"""

import numpy as np

from pprint import pprint
from scipy.integrate import dblquad

"""Useful Snippets"""

def RMatrix(theta):
    # theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def PinMatrix(matrix, right=True, lower=True):
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
    return PinMatrix(matrix, right, lower) == True

"""# 시그널 클래스 정의
시그널 클래스 (Signal) 가 정의된 이후에 (아날로그 시그널임), ```Signal``` 클래스를 상속한 클래스만이 이후에 시그널로 쓰일 수 있음. (이외에는 전부 AssertionError 로 Assert)
"""

class Signal:
    def GetPDF(self, X,Y, x0, y0, S):
        if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
            if X.size == Y.size:
                print("The signal is not defined. (this is pure class) Returning zero signal for all")
                return np.zeros(X.size)
            return 0
    def GetSignal(self, x, y, amplitude=100):
        return np.zeros(x.shape)

"""## GaussianSignal 정의
가우시안 분포를 기준으로 한 시그널

### 함수 설명
1. ``` GetStaticPDF ``` 수학적 정의인 StaticPDF 를 정의.
    * 객체 내 변수를 전혀 쓰지 않고, 함수의 변수만 씀.
2. ``` GetPDF ```       객체 내에 저장된 변수를 활용하여 StaticPDF 를 실제 PDF 로 변환
3. ``` GetSignal ```    실제 PDF 에 전체 적분값을 (amplitude) 를 곱하여 실제 시그널로 변환.

### 2D 가우시안
다음 세 값으로 정의됨.
1. 전체 적분값
2. 수직분산 (```sx, sy```)
3. 분산기준각 (```sdeg``` 또는 ```srad```, 둘 다 정의되면 ```sdeg```이 우선)

"""

class GaussianSignal(Signal):
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
        return np.matmul(RMatrix(srad), np.array([[sx,0],[0,sy]]))
    def GetStaticPDF(self, X,Y, x0, y0, S):
        X1 = X-x0
        Y1 = Y-y0
        Si = np.linalg.inv(S)
        X_ = Si[0][0] * X1 + Si[0][1] * Y1
        Y_ = Si[1][0] * X1 + Si[1][1] * Y1
        return np.exp(-0.5*(X_**2+Y_**2))/(2*np.pi*np.linalg.det(S))
    def GetPDF(self, x, y):
        return self.GetStaticPDF(x, y, self.x0, self.y0, self.S)
    def GetSignal(self, x, y):
        return self.amplitude * self.GetPDF(x,y)

"""# Digitizer 정의
Digitizer : 시그널을 검출기의 신호로 변환하는 것.

## 초기화

처음 정의할 때 필요한 것
다음 둘 중 하나만 하면 됨. (둘 다 하면, 1번이 인식됨. 필요시 변수 이름을 특정하여 적당한 ```__init__``` 의 변수만 지정 할 것.)
1. 최소, 최대 x,y 값과 양자화수
    * x최소값, x최대값, x칸수, y최소값, y최대값, y칸수
2. meshgrid 를 통해 생성한 X 와 Y 위치 

## 작동요소

1. ```InjectSignal(signal)``` 변수가 모두 들어가서 적절히 정의가 된 signal 을 주입하면, Digitizer 내의 signals 변수에 리스팅된다.
2. ```GetAnalogSignal(X,Y)``` (```X,Y``` 는 meshgrid 를 통해 형성된 공간좌표값) 을 하면, ```X,Y``` 로 주어진 공간에서의 아날로그 신호를 송출한다.
3. ```GetDigitalSignal_Old()``` 아날로그 시그널을 기준으로, ```Digitizer``` 에 정의된 디지털 그리드에 따라 해당 구역 내에 정의된 시그널 크기를 낸다. (그리드 센터의 값을 구한 다음에 면적을 곱하는 방식)
    * ```GetDigitalSignal()``` 그리드 내부를 전부 적분하는 방식  
      신호가 매우 날카로워서 그리드 센터에 신호가 없을 때의 문제를 해결하기 위함. 그러나 *느림*.
4. ```GetDigitizedSignal()``` 디지털 시그널을 기준으로, ```Digitizer``` 의 디지털 그리드에, 각각의 시그널이 ```on``` 인지 ```off```인지 판단한다
"""

class Digitizer:
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
        assert False
    def __SpanningBin(self, xmin, xmax, xn, ymin, ymax, yn):
        X, Y = np.meshgrid(np.linspace(xmin, xmax, xn+1), np.linspace(ymin, ymax, yn+1))
        self.__SpanningMeshgrid(X,Y)
    def __SpanningMeshgrid(self, X,Y):
        assert X.shape == Y.shape
        self.X = X
        self.Y = Y
        self.centerX = 0.5*(X[:,:-1]+X[:,1:])[:-1,:]
        self.centerY = 0.5*(Y[:-1,:]+Y[1:,:])[:,:-1]
    def InjectSignal(self, signal):
        assert isinstance(signal, Signal)
        # self.AnalogSignal = self.AnalogSignal + signal.GetSignal(self.X,self.Y)
        self.signals.append(signal)
    def ClearSignal(self):
        self.signals.clear()
    def GetAnalogSignal(self, X, Y):
        if isinstance(X,np.ndarray) or isinstance(Y,np.ndarray):
            assert X.shape == Y.shape
            AnalogSignal = np.zeros(X.shape)
        else:
            AnalogSignal = 0
        for signal in self.signals:
            AnalogSignal += signal.GetSignal(X,Y)
        # print("Total number of signal : %d"%(len(self.signals)) )
        return AnalogSignal
    def GetDigitalSignal_Old(self):
        centerSignal_ = self.GetAnalogSignal(self.centerX, self.centerY)
        centerSignal  = centerSignal_ * (PinMatrix(self.X) - PinMatrix(self.X, False, True))[1:-1,1:-1] * (PinMatrix(self.Y) - PinMatrix(self.Y, True, False))[1:-1,1:-1]
        return centerSignal
    def GetDigitalSignal_List(self):
        # OnPixels      = self.GetAnalogSignal(self.centerX, self.centerY)>0.1
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
            ansZ.append(dblquad(lambda x,y : self.GetAnalogSignal(x,y), ymin[i], ymax[i], xmin[i], xmax[i])[0]) # dblquad 를 할 때, 람다 수식 순서와 적분구간의 표기가 반대...? 라는 이야기가 있어서 바꿨는데
            ansX.append(X[i])
            ansY.append(Y[i])
            # print("Integral from [x,y]=[%d,%d] -> %.1f"%(xmin[i], ymin[i], ansZ[i]))
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
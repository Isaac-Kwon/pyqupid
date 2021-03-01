import numpy as np
from .Generator  import Signal, GaussianSignal, Digitizer # Simulator
from .Clustering import EventClustering, Pixel           # Clustering Algorithm
from .MonteCarloGenerator import GaussianSignalISOMonteCarlo

from .Grapher import PrintTFGrid

from random import random, seed

"""
Module: Simulator
"""

# Geometrical Constant
ALPIDEpitchx = 29.24e-3 # mm
ALPIDEpitchy = 26.88e-3 # mm

pitchx = 30./30. # Default : 1.
pitchy = 30./30. # Default : 1.

# Other constant
"""
number of electron
5.486 MeV / (786.5kJ/1mol) = 672763.593423027529918973 ~= 672764
"""
ne_alpha = 672764 # 5.44 MeV / Silicon's 1st ionization energy

class ClusterSimulation():
    def __init__(self, detector=None):
        if detector == None:
            self.detector = Digitizer(-16, 16, 32, -16, 16, 32)
        else:
            assert isinstance(detector, Digitizer)
            self.detector = detector
    def Analysis(self, pixlist):
        ec = EventClustering(pixlist)
        ec.ClusterAny()
        self.gridlist = list()
        for cluster in ec.CandidateClusters:
            x,y = cluster.GetAverage()
            cluster.ShiftBase(-int(x),-int(y))
            cluster.ShiftBase(int(self.detector.X.shape[1]/2)+1, int(self.detector.X.shape[0]/2)+1)
            grid = np.zeros(self.detector.X.shape, dtype="bool")
            grid.fill(False)
            for pixel in cluster:
                grid[pixel.y,pixel.x] = True
            self.gridlist.append(grid)
    def Clear(self):
        self.detector.ClearSignal()
        self.gridlist.clear()
        return None

class SingleGISOParticleSimulation(ClusterSimulation):
    def __init__(self, detector=None, sigrange=(0, ne_alpha), sigdev=0.83):
        super(SingleGISOParticleSimulation, self).__init__(detector=detector)
        self.mc = GaussianSignalISOMonteCarlo((-0.5, 0.5),
                                              (-0.5, 0.5),
                                              sigrange=sigrange, sigdev=sigdev)
        self.i=0
    def Execute(self):
        self.detector.InjectSignal(self.mc.GenerateSignal())
        fX, fY = self.detector.GetDigitizedSignal_List()
        pixlist = list()
        for i in range(fX.size):
            pixlist.append(Pixel(x=fX[i], y=fY[i]))
        self.Analysis(pixlist)
        self.i=self.i+1
        return None
    def Record(self):
        # with open(self.filename, 'a') as file:
        #     for grid in self.gridlist:
        #         result = '%d\t%.2f\t%.2f\t%d\t%d'%(self.i, self.detector.signals[0].x0, self.detector.signals[0].y0, self.detector.signals[0].amplitude, np.sum(np.sum(grid)))
        #         file.write(result+"\n")
        #         print(result)
        #         PrintTFGrid(grid)
        return None

class DoubleGISOParticleSimulation(ClusterSimulation):
    def __init__(self, detector=None, sigrange=(0, ne_alpha), sigdev=0.83, sig2posrange=(-8,8)):
        super(DoubleGISOParticleSimulation, self).__init__(detector=detector)
        self.mc1 = GaussianSignalISOMonteCarlo((-0.5, 0.5),
                                              (-0.5, 0.5),
                                              sigrange=sigrange, sigdev=sigdev)
        self.mc2 = GaussianSignalISOMonteCarlo((sig2posrange[0], sig2posrange[1]),(sig2posrange[0], sig2posrange[1]),
                                              sigrange=sigrange, sigdev=sigdev)
        self.i=0
    def Execute(self):
        self.detector.InjectSignal(self.mc1.GenerateSignal())
        self.detector.InjectSignal(self.mc2.GenerateSignal())
        fX, fY = self.detector.GetDigitizedSignal_List()
        pixlist = list()
        for i in range(fX.size):
            pixlist.append(Pixel(x=fX[i], y=fY[i]))
        self.Analysis(pixlist)
        self.i=self.i+1
        return None
    def Record(self):
        return None

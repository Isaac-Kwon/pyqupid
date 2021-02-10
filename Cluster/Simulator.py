import numpy as np
from .Generator  import Signal, GaussianSignal, Digitizer # Simulator
from .Clustering import EventClustering, Pixel           # Clustering Algorithm
from .MonteCarloGenerator import GaussianSignalISOMonteCarlo
import logging

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

class SingleGISOParticleSimulation(ClusterSimulation):
    def __init__(self, detector=None, sigrange=(0, ne_alpha), sigdev=0.83):
        super(SingleGISOParticleSimulation, self).__init__(detector=detector)
        self.mc = GaussianSignalISOMonteCarlo((-0.5, 0.5),
                                              (-0.5, 0.5),
                                              sigrange=sigrange, sigdev=sigdev)
        self.filename = 'text.txt'
        self.i=0
        with open(self.filename, 'w') as file:
            pass
    def Execute(self):
        self.detector.InjectSignal(self.mc.GenerateSignal())
        fX, fY = self.detector.GetDigitizedSignal_List()
        pixlist = list()
        for i in range(fX.size):
            pixlist.append(Pixel(x=fX[i], y=fY[i]))
        self.Analysis(pixlist)
        self.i=self.i+1
    def Record(self):
        with open(self.filename, 'a') as file:
            for grid in self.gridlist:
                result = '%d\t%.2f\t%.2f\t%d\t%d'%(self.i, self.detector.signals[0].x0, self.detector.signals[0].y0, self.detector.signals[0].amplitude, np.sum(np.sum(grid)))
                file.write(result+"\n")
                print(result)
                PrintTFGrid(grid)
    def Clear(self):
        self.detector.ClearSignal()
        self.gridlist.clear()
        return None

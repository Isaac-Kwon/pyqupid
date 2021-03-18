import sys

try:
    from qupid.Generator import Digitizer
    from qupid.Generator import GaussianSignal
    from qupid.Simulator import ClusterSimulation
    from qupid.Grapher   import PrintTFGrid
    from qupid.Tools     import ShrinkTFArray
    from qupid.Clustering import Pixel 
    from qupid.MonteCarloGenerator import GaussianSignalISOMonteCarlo
except ModuleNotFoundError:
    from os.path import realpath
    Repopath = realpath(__file__ + "/../../")
    sys.path.append(Repopath)
    from qupid.Generator import Digitizer
    from qupid.Generator import GaussianSignal
    from qupid.Simulator import ClusterSimulation
    from qupid.Grapher   import PrintTFGrid
    from qupid.Tools     import ShrinkTFArray
    from qupid.Clustering import Pixel 
    from qupid.MonteCarloGenerator import GaussianSignalISOMonteCarlo

from pandas import DataFrame
import numpy as np
from random import seed
from argparse import ArgumentParser

from random import random

import matplotlib.pyplot as plt

class CalibrationSingleExp(ClusterSimulation):#ne_alpha  : 540,272.98 = for 1.93958MeV 3.61eV/1e
    def __init__(self, detector=None, sigrange=(0, 540273), devrange=(0.75,0.91)):
        super(CalibrationSingleExp, self).__init__(detector=detector)
        self.mc = GaussianSignalISOMonteCarlo((-0.5, 0.5),
                                              (-0.5, 0.5),
                                              sigrange=sigrange, devrange=devrange)
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
        return None


def CalibrationRunFix(ntime = 10):
    df = DataFrame()
    print("===================")
    detector = Digitizer(-10, 10, 20, -10, 10, 20, threshold=250)
    exp = CalibrationSingleExp(detector=detector, sigrange=(540273, 540273), devrange=(0.83, 0.83))
    for i in range(ntime):
        exp.Execute()
        print("Signal Characteristics")
        print("i:%d \t x: %.2f \t y: %.2f \t amp: %d" %(i, detector.signals[0].x0, detector.signals[0].y0, detector.signals[0].amplitude))
        if len(exp.gridlist)==0:
            fired = None
            fnpix = 0
            print("None")
        else:
            fired = ShrinkTFArray(exp.gridlist[0])
            fnpix = np.sum(fired==True)
            PrintTFGrid(fired, truetext='O', falsetext='.')
        print("npixel=%d" %(fnpix))
        print("===================")
        stretch = fired.reshape((fired.size,))
        df1 = DataFrame({
            "i": i,
            "data": [fired],
            "x0": [detector.signals[0].x0],
            "y0": [detector.signals[0].y0],
            "sigdev" : [detector.signals[0].S[0,0]],
            "shapex": [fired.shape[1]],
            "shapey": [fired.shape[0]],
            "shapen": [sum(v<<i for i, v in enumerate(stretch[::-1]))],
            "amp": [detector.signals[0].amplitude],
            "npix": [fnpix]
        })
        df = df.append(df1, ignore_index=True)
        exp.Clear()
    return df

if __name__ == "__main__":
    parser = ArgumentParser(description='Signal Generation for Calibration Run')
    parser.add_argument('-r', dest='seed', help='Random Seed', type=int, default=0)
    parser.add_argument('-n', dest='nevent', help='Number of Event', type=int, default=10)
    parser.add_argument('-f', dest='filename', help='Filename', type=str, default="single.pkl")
    args = parser.parse_args()
    print("""
        Single Particle Simalation
        filename: %s
        random seed: %d
        number of event: %d
        """%(args.filename, args.seed, args.nevent)
    )
    seed(args.seed)
    df = CalibrationRun(args.nevent)
    df.to_pickle(args.filename)
    # dff = CalibrationRun(20)
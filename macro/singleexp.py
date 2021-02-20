import sys

try:
    from qupid.Generator import Digitizer
    from qupid.Simulator import ne_alpha
    from qupid.Simulator import SingleGISOParticleSimulation
    from qupid.Grapher   import PrintTFGrid
except ModuleNotFoundError:
    from os.path import realpath
    Repopath = realpath(__file__ + "/../../")
    sys.path.append(Repopath)
    from qupid.Generator import Digitizer
    from qupid.Simulator import ne_alpha
    from qupid.Simulator import SingleGISOParticleSimulation
    from qupid.Grapher   import PrintTFGrid


from pandas import DataFrame
import numpy as np
from random import seed

from argparse import ArgumentParser

def singleexp1(ntime = 10):
    df = DataFrame()
    print("===================")
    detector = Digitizer(-10, 10, 20, -10, 10, 20, threshold=250)
    exp = SingleGISOParticleSimulation(detector=detector, sigrange=(0, ne_alpha), sigdev=0.83)
    for i in range(ntime):
        exp.Execute()
        print("Signal Characteristics")
        print("i:%d \t x: %.2f \t y: %.2f \t amp: %d" %(i, detector.signals[0].x0, detector.signals[0].y0, detector.signals[0].amplitude))
        if len(exp.gridlist)==0:
            fired = None
            fnpix = 0
            print("None")
        else:
            fired = exp.gridlist[0]
            fnpix = np.sum(fired==True)
            PrintTFGrid(exp.gridlist[0], truetext='O', falsetext='.')
        print("npixel=%d" %(fnpix))
        print("===================")
        df1 = DataFrame({
            "i": i,
            "data": [fired],
            "x0": [detector.signals[0].x0],
            "y0": [detector.signals[0].y0],
            "amp": [detector.signals[0].amplitude],
            "npix": [fnpix]
        })
        df = df.append(df1)
        exp.Clear()
    return df


if __name__ == "__main__":
    parser = ArgumentParser(description='Signal Generation for single gaussian signal')
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
    df = singleexp1(args.nevent)
    df.to_pickle(args.filename)
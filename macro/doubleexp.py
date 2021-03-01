import sys

try:
    from qupid.Generator import Digitizer
    from qupid.Simulator import ne_alpha
    from qupid.Simulator import DoubleGISOParticleSimulation
    from qupid.Grapher   import PrintTFGrid
    from qupid.Tools     import ShrinkTFArray
except ModuleNotFoundError:
    from os.path import realpath
    Repopath = realpath(__file__ + "/../../")
    sys.path.append(Repopath)
    from qupid.Generator import Digitizer
    from qupid.Simulator import ne_alpha
    from qupid.Simulator import DoubleGISOParticleSimulation
    from qupid.Grapher   import PrintTFGrid
    from qupid.Tools     import ShrinkTFArray

from pandas import DataFrame
import numpy as np
from random import seed
from argparse import ArgumentParser

def doubleexp1(ntime = 10):
    df = DataFrame()
    print("===================")
    detector = Digitizer(-20, 20, 40, -20, 20, 40, threshold=250)
    exp = DoubleGISOParticleSimulation(detector=detector, sigrange=(0, ne_alpha), sigdev=0.83)
    for i in range(ntime):
        exp.Execute()
        print("Signal Characteristics")
        print("i:%d" %(i))
        print("signal 1 - x: %.2f \t y: %.2f \t amp: %d" %(detector.signals[0].x0, detector.signals[0].y0, detector.signals[0].amplitude))
        print("signal 2 - x: %.2f \t y: %.2f \t amp: %d" %(detector.signals[1].x0, detector.signals[1].y0, detector.signals[1].amplitude))
        if len(exp.gridlist)==0:
            # fired = [np.array([[False]], dtype='bool'), np.array([[False]], dtype='bool')]
            # fnpix = [0, 0]
            print("None")
            exp.Clear()
            continue
        elif len(exp.gridlist)==2:
            print("Double Clustered")
            print("Whole Detector")
            # fired = [ShrinkTFArray(exp.gridlist[0]), ShrinkTFArray(exp.gridlist[1])]
            # fnpix = [np.sum(fired[0]==True), np.sum(fired[1]==True)]
            PrintTFGrid(detector.GetDigitizedSignal(), truetext='O', falsetext='.')
            exp.Clear()
            print("===================")
            continue
        else:
            PrintTFGrid(detector.GetDigitizedSignal(), truetext='O', falsetext='.')
            fired = ShrinkTFArray(exp.gridlist[0])
            fnpix = np.sum(fired==True)
            PrintTFGrid(fired, truetext='O', falsetext='.')
        print("npixel=%s" %(fnpix))
        print("===================")
        df1 = DataFrame({
            "i": i,
            "data": [fired],
            "x0": [detector.signals[0].x0],
            "y0": [detector.signals[0].y0],
            "x1": [detector.signals[1].x0],
            "y1": [detector.signals[1].y0],
            "amp0": [detector.signals[0].amplitude],
            "amp1": [detector.signals[1].amplitude],
            "npix": [fnpix]
        })
        df = df.append(df1, ignore_index=True)
        exp.Clear()
    return df


if __name__ == "__main__":
    parser = ArgumentParser(description='Double Generation for single gaussian signal')
    parser.add_argument('-r', dest='seed', help='Random Seed', type=int, default=0)
    parser.add_argument('-n', dest='nevent', help='Number of Event', type=int, default=10)
    parser.add_argument('-f', dest='filename', help='Filename', type=str, default="double.pkl")
    args = parser.parse_args()
    print("""
        Double Particle Simalation
        filename: %s
        random seed: %d
        number of event: %d
        """%(args.filename, args.seed, args.nevent)
    )
    seed(args.seed)
    df = doubleexp1(args.nevent)
    df.to_pickle(args.filename)
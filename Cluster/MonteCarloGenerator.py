from random import random
from .Generator import GaussianSignal

def RandomUniform(range_min, range_max):
    interval = range_max-range_min
    return range_min + interval * random()

# only uniform distribution acceptable with automatic implementation.
# for other pdf, it should be assigned as function in MonteCarlo.gendict["nameofvariable"] directly.

# will be updated... Gaussian distribution or others....

# MCSampleDict = {
#     "x0" : {
#         "range": (0,1)
#     },
#     "y0" : {
#         "range" : (0,2)
#     },
#     "amp0": {
#         "fix" : 0.8
#     }
# }

class MonteCarlo:
    def __init__(self, geninfo):
        assert isinstance(geninfo, dict)
        self.gendict = geninfo
        for key in self.gendict.keys():
            if "range" in geninfo[key].keys():
                range = geninfo[key]["range"]
                self.gendict[key]["func"] = (lambda vmin, vmax : lambda : RandomUniform(vmin, vmax))(range[0], range[1])
            elif "fix" in geninfo[key].keys():
                self.gendict[key]["func"] = (lambda val : lambda : val)(self.gendict[key]["fix"])
    def Generate(self):
        for key in self.gendict.keys():
            assert callable(self.gendict[key]["func"])
            self.gendict[key]["value"] = self.gendict[key]["func"]()


class GaussianSignalISOMonteCarlo(MonteCarlo):
    def __init__(self, xrange, yrange, sigrange, sigdev=0.83):
        geninfo = { # Setup monte-carlo generator base
            "x0":{
                "range": (xrange[0], xrange[1])
            },
            "y0":{
                "range": (yrange[0], yrange[1])
            },
            "amp0":{
                "range": (sigrange[0], sigrange[1])
            },
            "sigdevx0": {
                "fix"  : sigdev
            },
            "sigdevy0": {
                "fix"  : sigdev
            }
        }
        super(GaussianSignalISOMonteCarlo, self).__init__(geninfo)
    def GenerateSignal(self):
        self.Generate()
        return GaussianSignal(x0=self.gendict["x0"]["value"],
                              y0=self.gendict["y0"]["value"],
                              amplitude=self.gendict["amp0"]["value"],sx=self.gendict["sigdevx0"]["value"],
                              sy=self.gendict["sigdevy0"]["value"]
                              )

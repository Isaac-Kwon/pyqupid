import numpy as np

def pixcut(pixlist):
    cutlist=np.array(pixlist).reshape(32,32)
    falserow=np.sum(cutlist, axis=1) #가로
    falsecol=np.sum(cutlist, axis=0) #세로
    for i in range(31,-1,-1):
        if falserow[i]==0:
            cutlist=np.delete(cutlist,i,axis=0)
        if falsecol[i]==0:
            cutlist=np.delete(cutlist,i,axis=1)
    return list(cutlist.flatten())
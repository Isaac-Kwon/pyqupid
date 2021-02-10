# -*- coding: utf-8 -*-
"""
MODULE : Clustering 
"""

import numpy as np

# Pixel
class Pixel():
    """ 
        Class Pixel
        Included Characteristics of "Pixel Number for 2D information (X, Y)"
    """
    def __init__(self, x = -1, y = -1,
                 globalX = 0., globalY = 0., globalZ = 0.):
        """ 
            Pixel.__init__(self, x:int = -1, y:int = -1)
            > Initializor of Pixel.
        """
        self.x = x
        self.y = y
        self.globalX = globalX
        self.globalY = globalY
        self.globalZ = globalZ 
    def __eq__(self, other):
        """
            Pixel.__eq__(self:Pixel, other:Pixel) -> bool
            > return true when 2 pixel have same coordinate (in int value)

            true: Coordinates of 2 pixels are same
            false: Coordinates of 2 pizels are different
        """
        return (self.x == other.x) and (self.y == other.y)
    def __repr__(self):
        return "<Pixel (%d,%d)>"%(self.x, self.y)
    def IsNear(self, other, diagonal=False, verbose=False):
        """
            Pixel.IsNear(self:Pixel, other:[Pixel,Cluster], diagonal:bool, verbose:bool) -> bool
            > return true when pixel (self) contacts with other pixel or cluster (other)
            
            true: pixel(self) contacts parameter [pixel, cluster] (other).
            false: pixel(self) doesn't contact with parameter [pixel, cluster] (other).
            + false: pixel(self) is included with parameter cluster
            + false: pixel(self) is same with parameter pixel
        """
        if isinstance(other,Pixel):
            if not diagonal:
                a = (abs(self.x-other.x)<=1 and abs(self.y-other.y)<=1) and (not (abs(self.x-other.x)==1 and abs(self.y-other.y)==1))
            else:
                a = abs(self.x-other.x)<=1 and abs(self.y-other.y)<=1
            if a and verbose:
                print("%s is near %s" %(self, other))
                print("%d - %d = %d" %(self.x, other.x, self.x-other.x))
                print("%d - %d = %d" %(self.y, other.y, self.y-other.y))
            return a
        elif isinstance(other,Cluster):
            return Cluster.IsNear(other, self, diagonal=diagonal)
        else:
            raise TypeError("Contact/Nearness Judgement (of Pixel) is not valid for %s" %(type(other)))
    def IsInside(self, other):
        """
            Pixel.IsInside(self:Pixel, self:Cluster) -> bool
            > return true if Pixel(self) is included in Cluster(other)
            > Get from Cluster Class' method (Cluster.IsInclude) with reverse parameter

            true: Pixel(self) is included in Cluster(other)
            false: Pixel(self) is not included in Cluster(other)
        """
        if not isinstance(other,Cluster):
            raise TypeError("Pixel.IsInside is not valid for %s" %(type(other)))
        return Cluster.IsInclude(other,self)
    def GetGlobalPosition(self):
        return np.array([self.globalX, self.globalY, self.globalZ])


# Cluster
class Cluster(list):
    """
        Class Cluster, list class inherited.
        It should be list of pixels.
    """
    def countInclusion(self, other):
        """
            Cluster.countInclusion(self:Cluster, other:Cluster) -> int
            > return how many pixel is overlapped with Cluster(other)
        """
        if not isinstance(other, Cluster):
            TypeError("counting inclusion is valid for Cluster only")
        nInside = 0
        for i in range(len(self)):
            for j in range(len(other)):
                if self[i]==other[j]:
                    nInside += 1
        return nInside
    def IsInclude(self, other): #Object 
        """ 
            Cluster.IsInclude(self:Cluster, other:[Cluster or Pixel]) -> bool
            > Judge Inclusion of [pixel, cluster] (other) with cluster (self).
            True: self include all pixels of other
            False: self doesn't include all pixels of other
        """
        if isinstance(other, Cluster):
            """
                In Case of Cluster, count pixels overlapped with clusters.
                if number pixel overlapped is same with number of element in cluster, return true
            """
            nInside = self.countInclusion(other)
            if nInside == len(other):
                return True
            else:
                return False
        elif isinstance(other,Pixel):
            """
                In case of pixel, find same pixel in cluster. 
                If there's same pixel, return True.
                If not, return False.
            """
            for i in range(len(self)):
                if self[i] == other:
                    return True
            return False
    def IsInside(self, other):
        """
            Cluster.IsInside(self:Cluster, other:Cluster)
        """
        if not isinstance(other,Cluster):
            raise TypeError("Cluster.IsInside method is not valid for %s"%(type(other)))
        return Cluster.IsInclude(other,self)
    def IsPiled(self, other):
        """
            Cluster.IsPiled(self:Cluster, other:Cluster)
        """
        if not isinstance(other, Cluster):
            raise TypeError("Cluster.IsPile method is not valid for %s"%(type(other)))
        return self.countInclusion(other)>0
    def IsNear(self, other, diagonal=False):
        """
            Cluster.IsNear(self:Cluster, other:[Cluster or Pixel], nrange:int=1, diagonal:bool) -> bool 
            > Judge the contactness of pixels.
            true  :: when pixel is contacted with cluster
            false :: when pixel is not contacted, OR included with cluster.
        """
        if isinstance(other,Pixel): # judgement nearness with pixel
            if self.IsInclude(other):
                return False
            for pix in self:
                if pix.IsNear(other,diagonal=diagonal):
                    return True
            return False
        elif isinstance(other,Cluster): # judgement nearness with cluster
            if not (self.IsPiled(other) or self.IsInclude(other) or self.IsInside(other)):
                return False
            for pix1 in other:
                for pix2 in self:
                    if pix1.IsNear(pix2,diagonal=diagonal):
                        return True
            return False
        else: # In case of wrong type (![Pixel, Cluster]) raise error
            raise TypeError("Contact/Nearness Judgement (of Cluster) is not valid for %s." %(type(other)))
    def IsRelated(self, other, diagonal=False):
        """
            Cluster.IsRelated(self:Cluster, other:[Pixel,Cluster], diagonal:bool) -> bool
            > return true if not seperated
            true: IsInside or IsNear or IsInclude or IsPile with parameter cluster
        """
        if isinstance(other,Pixel):
            return self.IsNear(other, diagonal=diagonal) or self.IsInclude(other)
        elif isinstance(other,Cluster):
            return self.IsNear(other, diagonal=diagonal) or self.IsInside(other) or self.IsInclude(other) or self.IsPiled(other)
        else:
            raise TypeError("Cluster.IsRelated is not valid for %s"%(type(other)))
    def AppendPixel(self, pixel, IgnoreScattered = False, IgnoreDuplicate = False, diagonal = False):
        """
            Cluster.AppendPixel(self:Cluster, pixel:Pixel, IgnoreScattered:bool, IgnoreDuplicate:bool, diagonal:bool) -> bool
            > Append pixel to cluster. 
            > IgnoreScattered: (Def: False) Append pixel in force even if pixel is not near with cluster
            > IgnoreDuplicated: (Def: False) Append Pixel in force even if pixel is already in Cluster (same pixel in Cluster)
            True: Pixel is appended into Cluster
            False: Pixel is not appended into cluster due to some reason (duplicated, scattered)
        """
        if not isinstance(pixel, Pixel):
            raise TypeError("Cluster.AppendPixel is not valid for %s" %(type(pixel)))
        if not IgnoreScattered:
            if not self.IsNear(pixel, diagonal=diagonal) and not len(self) == 0:
                return False
        if not IgnoreDuplicate:
            if self.IsInclude(pixel):
                return False
        self.append(pixel)
        return True
    def CheckCluster(self, diagonal=False):
        """
            Cluster.CheckCluster(self:Cluster, diagonal:bool) -> bool
            > Check cluster whethter the cluster is gathered or not.
            true: all pixel in cluster is gathered in one
            false: there's 
        """
        for pix1 in self:
            for pix2 in self:
                if pix1.IsNear(pix2, diagonal=diagonal):
                    pass
                else:
                    return False
        return True
    def GetAverage(self):
        sumx = 0
        sumy = 0
        for pixel in self:
            sumx = sumx + pixel.x
            sumy = sumy + pixel.y
        return sumx/len(self), sumy/len(self)
    def ShiftBase(self, dx=0, dy=0):
        for pixel in self:
            pixel.x = pixel.x+dx
            pixel.y = pixel.y+dy


# Clustering the Event

class EventClustering():
    """
        Class EventClustering
    """
    def __init__(self, CandidatePixels = list()):
        """
            EventClustering.__init__(self:EventClustering, CandidatePixel:list)
        """
        self.CandidatePixels = CandidatePixels
        self.CandidateClusters = list()
    def __repr__(self):
        """
            EventClustering.__repr__(self:EventClustering)
        """
        return "Cluster: %s, Pixels: %s" %(self.CandidateClusters, self.CandidatePixels)
    def ClusterOne(self, diagonal=False):
        """
            EventClustering:ClusterOne(self:EventClustering, diagonal:bool)
        """
        if len(self.CandidatePixels) == 0:
            raise AssertionError("There's no element in candidate pixels")
        SPixel = self.CandidatePixels.pop() # Seed Pixel Candidate
        C0 = Cluster()
        self.CandidateClusters.append(C0)
        C0.AppendPixel(SPixel, diagonal=diagonal)
        nPixCand = -1
        Repeat = True
        while Repeat:
            if len(self.CandidatePixels) == nPixCand:
                break
            else:
                nPixCand = len(self.CandidatePixels)
            for DPixC in self.CandidatePixels: # Dummy Pixel Candicate
                if DPixC.IsNear(C0):
                    # print("%s is near %s" %(DPixC, C0))
                    C0.AppendPixel(self.CandidatePixels.pop(self.CandidatePixels.index(DPixC)))
                    break
        return C0
    def ClusterAny(self, diagonal=False):
        """
            EventClustering:ClusterAny(self:EventClustering, diagonal:bool)
        """
        while len(self.CandidatePixels)>=1:
            self.ClusterOne(diagonal=diagonal)
        return
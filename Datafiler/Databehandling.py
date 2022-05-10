
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, FancyArrow
import scipy.optimize as sc
from scipy.integrate import solve_ivp
import sympy as sp
import os
import math



"""
Funktion der deler et array af arrays op efter
"""
def divideBy(arrays,axis, start, end, num):
    out_Arrays = []

    delta = (end - start)/num
    divider = arrays[axis]
    for i in range(0,num):
        indexs = np.where(start+delta*i =< divider =<end+delta*i,range(len(divider)),np.empty(len(divider)))
        indexs = indexs[~(numpy.isnan(indexs))]
        out_array = []
        for array in arrays:
            out_array +=np.array(array[indexs])
        out_Arrays += [np.array(out_array)]

    return np.array(out_Arrays)

"""
Objekt til at gemme på værdier som gælder for hele undersøgelsen, dvs. vægte af lod, hvad fps på kameraet er, osv.


"""
#K
class diverseVærdier:
    fps = 60.0
    m_tung = 20.2*10**(-3)
    m_medium = 9.5*10**(-3)
    m_let = 5.1*10**(-3)
    m_magnet = 42.5*10**(-3)
    g = -9.82
    spoleOffset = 0
    rørOffset = 0




"""
Objekt til at hånterer en video/måling

(Subclass af undersøgelse objektet, hvilket betyder at den "arver" alle de værdier som gælder for hele undersøgelsen, f.eks. vægte af lod og fps)
"""
class Maaling(diverseVærdier):
    def __init__(self,file):
        self.file = file

        """
        Kode til at give lod den korrekte masse
        """
        if "let" in file:
            self.m_lod = self.m_let
        elif "medium" in file:
            self.m_lod = self.m_medium
        elif "tung" in file:
            self.m_lod = self.m_tung
        print(self.m_lod)
        """
        Offset mellem magnetens, og loddets position; vi mangler måske noget logik her til at tage højde for for hvilket lod vi bruger

        """

        if "Rør" in file:
            self.offset = self.rørOffset
        elif "Spole" in file:
            self.offset = self.spoleOffset




        """
        Loader data fra filen ind
        """
        self.rawT, self.yLod = np.loadtxt(self.file,skiprows=3,unpack = True)

        """
        Vi skaber en liste af målte tider ud fra vores fps
        """
        self.ts = np.array(range(0,len(self.yLod)))*(1/self.fps)




        """
        Vi finder magnetens position
        """

        self.ys = -self.yLod+self.offset
        self.ys_unc = np.zeros(len(self.ys))

        """
        Vi finder magnetens hastighed
        """
        self.vs = np.insert(np.diff(self.ys)/np.diff(self.ts),0,0, axis = 0)
        self.vs_unc = np.zeros(len(self.vs))
        """
        Og Acceleration
        """
        self.as = np.insert(np.diff(self.vs)/np.diff(self.ts),  0,0, axis = 0)
        self.as_unc = np.zeros(len(self.as))
        """
        Og endeligt udregner vi størrelsen på bremsekraften
        """

        self.Fs = self.as*(self.m_magnet+self.m_lod)+self.g*(self.m_lod-self.m_magnet)
        self.Fs_unc = np.zeros(len(self.Fs))

        return

"""
Objekt til håntering af alle målingerne på et af de ledende elementer
"""
class ledendeElement(diverseVærdier):
    def __init__(self):

        self.files = []
        self.maalinger = []

        self.ts = np.array([])
        self.ys = np.array([])
        self.vs = np.array([])
        self.as = np.array([])
        self.Fs = np.array([])

        self.ys_unc = np.array([])
        self.vs_unc = np.array([])
        self.as_unc = np.array([])
        self.Fs_unc = np.array([])

        return

    def addData(self,file):

        maaling = Maaling(file)

        self.ts = np.append(self.ts,maaling.ts)
        self.ys = np.append(self.ys,maaling.ys)
        self.vs = np.append(self.vs,maaling.vs)
        self.as = np.append(self.as,maaling.as)
        self.Fs = np.append(self.Fs,maaling.Fs)

        self.ys_unc = np.append(self.ys_unc,maaling.ys_unc)
        self.vs_unc = np.append(self.vs_unc,maaling.vs_unc)
        self.as_unc = np.append(self.as_unc,maaling.as_unc)
        self.Fs_unc = np.append(self.Fs_unc,maaling.Fs_unc)


        self.files += [file]
        self.maalinger += [maaling]
        return

"""
"""

dataFolder = "."




file = "Rør, let, bund 1.txt"

fig, ax = plt.subplots(1,1)

maaling = Maaling(file)

ax.plot(maaling.ts, maaling.vs, marker = ".", ls = "None")
ax.plot(maaling.ys, maaling.Fss, marker = ".", ls = "None")


plt.show()

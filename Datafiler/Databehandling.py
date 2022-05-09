
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
Objekt til at gemme på værdier som gælder for hele undersøgelsen, dvs. vægte af lod, hvad fps på kameraet er, osv.


"""
#K
class undersøgelse:
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
class Maaling(undersøgelse):
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

        self.yMagnet = -self.yLod+self.offset


        """
        Vi finder magnetens hastighed
        """
        self.vMagnet = np.insert(np.diff(self.yMagnet)/np.diff(self.ts),0,0, axis = 0)
        """
        Og Acceleration
        """
        self.aMagnet = np.insert(np.diff(self.vMagnet)/np.diff(self.ts),  0,0, axis = 0)

        """
        Og endeligt udregner vi størrelsen på bremsekraften
        """

        self.Fbrems = self.aMagnet*(self.m_magnet+self.m_lod)+self.g*(self.m_lod-self.m_magnet)


        return


file = "Rør, let, bund 1.txt"

fig, ax = plt.subplots(1,1)

maaling = Maaling(file)

ax.plot(maaling.ts, maaling.vMagnet, marker = ".", ls = "None")
ax.plot(maaling.ts, maaling.aMagnet, marker = ".", ls = "None")


plt.show()

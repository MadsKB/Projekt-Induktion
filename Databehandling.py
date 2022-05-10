
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

    Empt = np.empty(len(divider))
    Empt[:] = np.nan

    for i in range(0,num):
        print("Start",(start+delta*i))
        print("End", (delta*(i+1)))
        indexs = np.where(np.logical_and((start+delta*i) <= divider, divider < (delta*(i+1))),np.array(list(range(len(divider))),dtype = np.int64),Empt)
        #print(indexs)
        indexs = indexs[~(np.isnan(indexs))]
        #print(indexs)
        out_array = []
        for array in arrays:
            #print(array)
            out_array += [array[indexs.astype(np.int64)]]
        out_Arrays += [out_array]
    print(out_Arrays)
    return out_Arrays

"""
Udregner chi^2 af fit (chi^2_min)
"""

def chi2_min(fit,x,y,parms,yerr):
    return np.sum((y-fit(x,*parms))/yerr)



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

        self.As = np.insert(np.diff(self.vs)/np.diff(self.ts),  0,0, axis = 0)
        self.As_unc = np.zeros(len(self.As))
        """
        Og endeligt udregner vi størrelsen på bremsekraften
        """

        self.Fs = self.As*(self.m_magnet+self.m_lod)+self.g*(self.m_lod-self.m_magnet)
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
        self.As = np.array([])
        self.Fs = np.array([])

        self.ys_unc = np.array([])
        self.vs_unc = np.array([])
        self.As_unc = np.array([])
        self.Fs_unc = np.array([])

        return

    def addData(self,file):

        maaling = Maaling(file)

        self.ts = np.append(self.ts,maaling.ts)
        self.ys = np.append(self.ys,maaling.ys)
        self.vs = np.append(self.vs,maaling.vs)
        self.As = np.append(self.As,maaling.As)
        self.Fs = np.append(self.Fs,maaling.Fs)

        self.ys_unc = np.append(self.ys_unc,maaling.ys_unc)
        self.vs_unc = np.append(self.vs_unc,maaling.vs_unc)
        self.As_unc = np.append(self.As_unc,maaling.As_unc)
        self.Fs_unc = np.append(self.Fs_unc,maaling.Fs_unc)


        self.files += [file]
        self.maalinger += [maaling]
        return

"""
"""
class undersøgelse:
    def __init__(self, dataFolder):
        self.rør = ledendeElement()
        self.Spole = ledendeElement()
        self.SpoleLukket = ledendeElement()

        self.rørFiles           = []
        self.SpoleFiles         = []
        self.SpoleLukketFiles   = []


        allFiles = os.listdir(dataFolder)

        for file in allFiles:

            #problemer med file Spole, let, bund 1.txt (Nogle af y-værdierne i den manglet, dette fikset jeg ved at erstatte disse med 0)
            if "Rør" in file:
                self.rørFiles += [file]
                self.rør.addData(dataFolder+file)
            elif "Spole lukket" in file:
                self.SpoleLukketFiles += [dataFolder+file]
                self.SpoleLukket.addData(dataFolder+file)
            elif "Spole" in file:
                self.SpoleFiles += [dataFolder+file]
                self.Spole.addData(dataFolder+file)



        return




und = undersøgelse("./Datafiler/")


fig, ax = plt.subplots(1,1)


vedMunding = divideBy([und.rør.ys,und.rør.vs,und.rør.Fs],0,0,0.1,10)[0]
#print(len(vedMunding))

ax.plot(vedMunding[0],vedMunding[2],ls = "None", marker = ".")
ax.set_ylabel(r"$F_{brems} \quad [N]$")
ax.set_xlabel(r"$v \quad [m/s]$")
ax.grid()



plt.show()

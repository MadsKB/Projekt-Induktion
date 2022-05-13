
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, FancyArrow
import scipy.optimize as sc
import scipy.stats as ss
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
        print("kage", i)
        print(delta)
        #print("Start",(start+delta*i))
        #print("End", (delta*(i+1)))
        indexs = np.where(((start+delta*i) <= divider)*(divider < (start+delta*(i+1))),np.array(list(range(len(divider))),dtype = np.int64),Empt)
        #print(indexs)
        indexs = indexs[~(np.isnan(indexs))]
        #print(indexs)
        print(len(indexs))
        out_array = []
        for array in arrays:
            #print(array)
            out_array += [array[indexs.astype(np.int64)]]
        out_Arrays += [out_array]

    return out_Arrays

"""
Udregner chi^2 af fit (chi^2_min)
"""

def chi2_min(fit,x,y,parms,yerr):
    return np.sum((y-fit(x,*parms))/yerr)

def calcP(parms,ys,chi):
    return ss.chi2.sf(chi,len(ys)-len(parms))

def fitTo(fit,x,y,yerr, P0):
    parms, pcov = sc.curve_fit(fit, x , y ,p0 = P0, sigma = yerr, absolute_sigma = True, maxfev = 1000000)
    return parms, pcov, chi2_min(fit,x,y,parms,yerr), calcP(parms,y,chi2_min(fit,x,y,parms,yerr))





"""
Objekt til at gemme på værdier som gælder for hele undersøgelsen, dvs. vægte af lod, hvad fps på kameraet er, osv.


"""
#K
class diverseVærdier:
    fps = 240.0
    m_tung = 20.2*10**(-3)
    m_medium = 9.5*10**(-3)
    m_let = 5.1*10**(-3)
    m_magnet = 42.5*10**(-3)
    g = -9.6
    spoleOffset = np.mean(np.array([1.323263E-1, 1.174768E-1,1.348158E-1]))
    rørOffset = np.mean(np.array([2.635928E-2,3.777756E-2,-1.330933E-2]))
    LSpole = 0.191
    LRør = 0.251
    lowLim = 0.4
    highLim = 0.4
    midtPoint = 1/2
    errdataFile = "Induktionsbremse fejltest.txt"

    t,ylod1,ylod2,ylod3,ylod4 = np.loadtxt(errdataFile, unpack = True, usecols=(0,1,2,3,4), skiprows=3)
    yerr = sum((abs(ylod1-ylod2)+abs(ylod3-ylod4))/2)/len(ylod1)



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
            self.L = self.LRør
        elif "Spole" in file:
            self.offset = self.spoleOffset
            self.L = self.LSpole

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
        self.ys_unc = np.full(len(self.ys), self.yerr)

        #self.ys_unc = np.zeros(len(self.ys))

        """
        Vi finder magnetens hastighed
        """
        self.vs = np.gradient(self.ys,self.ts)
        #self.vs = np.insert(np.diff(self.ys)/np.diff(self.ts),0,0, axis = 0)
        self.vs_unc = np.zeros(len(self.vs))
        """
        Og Acceleration
        """
        self.As = np.gradient(self.vs,self.ts)
        #self.As = np.insert(np.diff(self.vs)/np.diff(self.ts),  0,0, axis = 0)
        self.As_unc = np.zeros(len(self.As))
        """
        Og endeligt udregner vi størrelsen på bremsekraften
        """
        self.Fs = self.As*(self.m_magnet+self.m_lod)+self.g*(self.m_lod-self.m_magnet)
        self.Fs_unc = np.zeros(len(self.Fs))





        self.upperBound = self.highLim
        self.lowerBound = self.lowLim


        self.inLeder = divideBy([self.ys,self.ys_unc,self.ts],0,-self.L*(self.midtPoint+self.upperBound),-self.L*(self.midtPoint-self.lowerBound),4)

        fitter = lambda t, a, v0,y0: 1/2*a*t**2+v0*t+y0

        self.ys_inLeder     = np.array([])
        self.ys_unc_inLeder = np.array([])
        self.ts_inLeder     = np.array([])

        self.a_in   = np.array([])
        self.v0_in  = np.array([])
        self.y0_in  = np.array([])

        self.v0_in_unc  = np.array([])
        self.a_in_unc   = np.array([])

        self.F_in     = np.array([])
        self.F_in_unc = np.array([])
        print(self.file)
        for i,mål in enumerate(self.inLeder):
            print("På ", i, len(mål[0]))
            if len(mål[0]) <= 3:
                continue
            self.ys_inLeder = np.append(self.ys_inLeder,mål[0])
            self.ys_unc_inLeder = np.append(self.ys_unc_inLeder,mål[1])
            self.ts_inLeder = np.append(self.ts_inLeder,mål[-1])




        #parms, pcov = sc.curve_fit(fitter, self.ts_inLeder , self.ys_inLeder , sigma = self.ys_unc_inLeder, absolute_sigma = True, maxfev = 1000000)
        #try:

            #sigma = self.ys_unc_inLeder
            parms, pcov = sc.curve_fit(fitter, mål[-1] , mål[0],p0 = [self.g,-1,0], sigma= mål[1] , absolute_sigma = True, maxfev = 1000000)

            self.a_in   = np.append(self.a_in , parms[0])
            self.v0_in  = np.append(self.v0_in, parms[1])
            self.y0_in  = np.append(self.y0_in, parms[2])
            self.v0_in_unc  = np.append(self.v0_in_unc, np.sqrt(pcov[1,1]))
            self.a_in_unc   = np.append(self.a_in_unc , np.sqrt(pcov[0,0]))

            self.F_in       = np.append( self.F_in    ,parms[0]*(self.m_magnet+self.m_lod)+self.g*(self.m_lod-self.m_magnet))
            self.F_in_unc   = np.append( self.F_in_unc,np.sqrt(pcov[0,0])*abs(self.m_magnet+self.m_lod))


        #except:
            #print("Oi")
            #self.a_in = np.nan
            #self.v0_in = np.nan
            #self.v0_in_unc = np.nan
            #
            #self.F_in = np.nan
            #self.F_in_unc = np.nan

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
        self.newVs = np.array([])
        self.newFs = np.array([])

        self.ys_unc = np.array([])
        self.vs_unc = np.array([])
        self.As_unc = np.array([])
        self.Fs_unc = np.array([])

        self.newVs_unc = np.array([])
        self.newFs_unc = np.array([])

        return

    def addData(self,file):

        maaling = Maaling(file)

        self.ts = np.append(self.ts,maaling.ts)
        self.ys = np.append(self.ys,maaling.ys)
        self.vs = np.append(self.vs,maaling.vs)
        self.As = np.append(self.As,maaling.As)
        self.Fs = np.append(self.Fs,maaling.Fs)

        self.newVs = np.append(self.newVs,maaling.v0_in)
        self.newFs = np.append(self.newFs,maaling.F_in)

        self.ys_unc = np.append(self.ys_unc,maaling.ys_unc)
        self.vs_unc = np.append(self.vs_unc,maaling.vs_unc)
        self.As_unc = np.append(self.As_unc,maaling.As_unc)
        self.Fs_unc = np.append(self.Fs_unc,maaling.Fs_unc)

        self.newVs_unc = np.append(self.newVs,maaling.v0_in_unc)
        self.newFs_unc = np.append(self.newFs,maaling.F_in_unc)


        self.files += [file]
        self.maalinger += [maaling]
        return

"""
"""
class undersøgelse(diverseVærdier):
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

midtIrør = -und.LRør*(1/2-0.3)
Mål = 1
PFactor = 0.4

vedMundingRør = divideBy([und.rør.maalinger[Mål].ys,und.rør.maalinger[Mål].ts,und.rør.maalinger[Mål].vs,und.rør.maalinger[Mål].Fs,und.rør.maalinger[Mål].vs_unc,und.rør.maalinger[Mål].Fs_unc],0,-und.LRør*(1/2+PFactor),-und.LRør*(1/2-PFactor),1)[0]



#vedMundingRør = divideBy([und.rør.ys,und.rør.vs,und.rør.Fs,und.rør.vs_unc,und.rør.Fs_unc],0,-und.LRør*(1/2+0.15),-und.LRør*(1/2-0.15),1)[0]
vedMundingSpole = divideBy([und.Spole.maalinger[Mål].ys,und.Spole.maalinger[Mål].ts,und.Spole.maalinger[Mål].vs,und.Spole.maalinger[Mål].Fs,und.Spole.maalinger[Mål].vs_unc,und.rør.maalinger[Mål].Fs_unc],0,-und.LSpole*(1/2+PFactor),-und.LSpole*(1/2-PFactor),1)[0]
vedMundingSpoleLukket = divideBy([und.SpoleLukket.ys,und.SpoleLukket.vs,und.SpoleLukket.Fs,und.SpoleLukket.vs_unc,und.SpoleLukket.Fs_unc],0,-und.LSpole*(1/2+0.15),-und.LSpole*(1/2-0.15),1)[0]
#print(len(vedMunding))
#ax.plot(und.rør.maalinger[0].ts,und.rør.maalinger[0].Fs,ls = "None", marker = ".",label = "rør",alpha = 0.7)
#ax.plot(und.Spole.vs,und.Spole.Fs,ls = "None", marker = ".", label = "Spole",alpha = 0.7)
#ax.plot(und.SpoleLukket.vs,und.SpoleLukket.Fs,ls = "None", marker = ".", label = "Spole lukket",alpha = 0.7)

#ax.errorbar(vedMundingRør[2],vedMundingRør[3],yerr = vedMundingRør[-1],xerr = vedMundingRør[-2],ls = "None", marker = ".", label = "Rå data Rør")
#ax.errorbar(vedMundingRør[1],vedMundingRør[0],yerr = vedMundingRør[-1],xerr = vedMundingRør[-2],ls = "None", marker = ".", label = "Rå data Rør")
#ax.errorbar(vedMundingSpole[1],vedMundingSpole[0],yerr = vedMundingSpole[-1],xerr = vedMundingSpole[-2],ls = "None", marker = ".", label = "Rå data Spole")
#ax.errorbar(vedMundingSpoleLukket[1],vedMundingSpoleLukket[2],yerr = vedMundingSpoleLukket[-1],xerr = vedMundingSpoleLukket[-2],ls = "None", marker = ".", label = "Rå data Spole lukket")
print(und.rør.newVs)
print(und.rør.newFs)

#ax.errorbar(vedMundingRør[1],vedMundingRør[0],yerr = vedMundingRør[-1],xerr = vedMundingRør[-2],ls = "None", marker = ".", label = "Rå data Rør")

#ax.errorbar(und.rør.maalinger[Mål].ts_inLeder,und.rør.maalinger[Mål].ys_inLeder,ls = "None", marker = ".", label = "Rå data Rør")
#fitted = und.rør.maalinger[Mål].a_in*1/2*np.power(und.rør.maalinger[Mål].ts_inLeder,2)+und.rør.maalinger[Mål].v0_in*und.rør.maalinger[Mål].ts_inLeder+und.rør.maalinger[Mål].y0_in
#ax.plot(und.rør.maalinger[Mål].ts_inLeder,fitted,ls = "--",marker = "None", label = "Fit til kons. acceleration")
print(und.rør.newVs_unc)

lin = lambda x,a: a*x

#ax.errorbar(und.rør.newVs,und.rør.newFs,xerr = und.rør.newVs_unc[0:len(und.rør.newVs)],yerr = und.rør.newFs_unc[0:len(und.rør.newVs)],ls = "None", marker = ".", label = "Data", elinewidth = 0.4)
#ax.errorbar(und.SpoleLukket.newVs,und.SpoleLukket.newFs,xerr = und.SpoleLukket.newVs_unc[0:len(und.SpoleLukket.newVs)],yerr = und.SpoleLukket.newFs_unc[0:len(und.SpoleLukket.newVs)],ls = "None", marker = ".", label = "Data", elinewidth = 0.4)
ax.errorbar(und.SpoleLukket.newVs,und.SpoleLukket.newFs,xerr = und.SpoleLukket.newVs_unc[0:len(und.SpoleLukket.newVs)],yerr = und.SpoleLukket.newFs_unc[0:len(und.SpoleLukket.newVs)],ls = "None", marker = ".", label = "Data", elinewidth = 0.4)


parms, pcov, chi2_min, P = fitTo(lin,und.SpoleLukket.newVs,und.SpoleLukket.newFs, und.SpoleLukket.newFs_unc[0:len(und.SpoleLukket.newVs)],[-10])
vs = np.linspace(-30,75,100)
print(chi2_min)


ax.plot(vs,parms[0]*vs,ls = "--",marker = "None", label = "Fit med lineær model; $F = k\cdot v$, \n k = {0:0.3f} \n P-værdi: {1:0.3f}".format(parms[0],P), lw = 1)


#ax.errorbar(und.Spole.newVs,und.Spole.newFs,xerr = und.Spole.newVs_unc[0:-1],yerr = und.Spole.newFs_unc[0:-1],ls = "None", marker = ".", label = "Rå data Spole")
#yerr = und.rør.newFs_unc,xerr = und.rør.newVs_unc

labelfontsize = 12
ax.set_title(r"Bremse kraft som funktion af hastighed i 'åben' spole" "\n (Indenfor $\pm 0.4 \cdot L_{spole}$ af centrum)")
ax.set_ylabel(r"$F_{brems} \quad [N]$", fontsize = labelfontsize)
ax.set_xlabel(r"$v \quad [m/s]$",       fontsize = labelfontsize)

#ax.set_ylabel(r"$y \quad [m]$")
#ax.set_xlabel(r"$ts \quad [s]$")

ax.grid()
ax.tick_params(labelsize = 10)
ax.legend()


plt.show()

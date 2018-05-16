# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import numpy.linalg
import scipy.sparse as sparse
import scipy.sparse.linalg
from math import *

import matplotlib.pyplot as pp



def qubitSpectrum(qubitParams, N, extFlux, extVoltage):

    step = 2*pi/N
    
    
    h = 6.626e-34;
    hbar = h/2/pi
    e = 1.6e-19;
    Fi0 = h/2/e;
      
#    L = qubitParam.L/cos(extFlux*pi/Fi0);
    C = qubitParams.Cq + qubitParams.Cg + qubitParams.Cx;
    
    #Ic =  qubitParams.Ic*abs(cos(pi*extFlux/Fi0))
    Ic =  qubitParams.Ic * abs(cos(pi*extFlux))

    Ej = (2 * Ic * Fi0 / 2) / (h*1e9) # GHz
    Ec = ((1 * e)**2 / (2 * C)) / (h*1e9) # GHz
    ng = extVoltage * qubitParams.Cx / (2 * e);
#    print('Ec = %e, Ej = %e, Ej/Ec = %f, E01 = %e,' % (Ec, Ej, Ej/Ec, sqrt(8*Ec*Ej) - Ec))
#    print(ng)

    
    
    phi = np.linspace(-pi, pi, N+1);
    phi = phi[0 : -1];
    
    def alpha(phi):
        return -4*Ec
    def beta(phi):
        return 0
    def gamma(phi):
       return -cos(phi)*Ej
    
    diagCentr = np.zeros([N], dtype='complex')
    diagUp = np.zeros([N], dtype='complex')
    diagDown = np.zeros([N], dtype='complex')
    
    for i in range(N):
        diagCentr[i] = gamma(phi[i])  - 2*alpha(phi[i])/(step*step)
        diagUp[i] = alpha(phi[i])/(step*step) + beta(phi[i])/2/step
        diagDown[i] = alpha(phi[i])/(step*step) - beta(phi[i])/2/step
    
    phasefactor = np.exp(1j*ng*pi)

  
    sm = sparse.diags([[np.conj(phasefactor)*diagUp[-1]], diagDown[1:], diagCentr, diagUp[0: -1], [phasefactor*diagDown[1]]], [-N + 1, -1, 0, 1, N -1])
    sm = sm.toarray();
    (ev, evec) = np.linalg.eigh(sm)
#    sparse.linalg.eigs(sm, 3, which='SM')
  
    return ev, phi, evec

class qubitParams:
    
    C = None
    Ic = None
    def __init__(self):
        self.Cq = 1.01e-13
        self.Ic = 30e-9
        self.Cx = 1e-16
        self.Cg = 1e-14


if __name__ == '__main__':
    qp = qubitParams()
    
#    fluxes = np.linspace(-3e-5, 3e-5, 0)
##    fluxes = [0]
#    evs = np.zeros([len(fluxes), 6])
#    
#    for i in range(len(fluxes)):
#        (ev, phis, evec) = qubitSpectrum(qp, 100, 0, fluxes[i])
#        
#        evs[i, :] = sorted(ev)
#    
    

    voltage = np.linspace(-5e-3, 5e-3, 500)
    evals = np.zeros([len(voltage), 5])
    
    for i in range(len(evals)):
        
        (ev, phis, evec) = qubitSpectrum(qp, 30, 0.35, voltage[i])
        ev = np.real(ev)
        ev.sort()
        evals[i] = ev[0:5] -ev[0]
        print('E01 = %e, E12 = %e '%(ev[1] - ev[0], ev[2] - 2*ev[1] + ev[0]) )

    pp.plot(voltage, evals)
    pp.xlabel('Vxy, V')
    pp.ylabel('E, GHz')
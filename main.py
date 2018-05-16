#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# разработать алгоритм управления квантовым резонатором, который бы переводил 
# квантовый кубит из начального в заданное состояние с максимальной точностью
# (loss -> 0)
#
# пример алгоритмов управления задаются функциями типа 
# dummyExternalDrive(time, qStart, qTgt)


from qutip import *
import numpy as np
import matplotlib.pyplot as pp
from qubitSpectr import *
import time
import scipy.fftpack as fft

# constants & params
e = 1
hbar = 1
hplank = hbar * 2 * pi
Fi0 = 2*pi*hbar/(2*e)

nmax = 5


qp = qubitParams()
Cx = qp.Cx
Cg = qp.Cg
Cq = qp.Cq
Zr = 50
extFlux = -.58
extVoltage = 0.0
omega = 6

Q = 1e3
T1 = 1e9

Cr = 1/(omega*1e9*Zr)
(ev, phis, evec) = qubitSpectrum(qp, 100, extFlux, extVoltage)
epsilon  = ev[1] - ev[0]
#print('Epsilon = %e, detuning = %.2f' % (epsilon, (epsilon - omega) / g))
g = Cg/(2*sqrt(Cq*Cr))*sqrt(omega*epsilon)

Aplus = tensor(create(nmax),qeye(2))
Aminus = tensor(destroy(nmax), qeye(2))
eField = Aminus + Aplus

SigmaX = tensor(qeye(nmax), sigmax())
SigmaZ = tensor(qeye(nmax), sigmaz())
SigmaP = tensor(qeye(nmax), sigmap())
SigmaN = tensor(qeye(nmax), sigmam())
SigmaPopulation = tensor(qeye(nmax), (sigmaz() + qeye(2)) / 2)


# default frequency

wDrive = epsilon
times = np.linspace(0, 100, 8000) # control period (8000 is 8ns)

# initial state

qubitStartState = fock(2, 1)
qubitStartState = qubitStartState.unit()
qubitStartState = qubitStartState*qubitStartState.dag()
resonatorStartState = fock(nmax, 0)
resonatorStartState = resonatorStartState*resonatorStartState.dag()

# target state

qubitTgtState = fock(2, 0)
qubitTgtState = qubitTgtState.unit()
qubitTgtState = qubitTgtState*qubitTgtState.dag()

# control strategies

def dummyExternalDrive(t, start, target):
    return 1e-5*sin(2*pi*t*wDrive)


def ramseyExternalDrive(t, start, target):
    tint = t
    ramseyDelay = 4
    pi4Time = 3
    if tint < pi4Time:
        return 0.1*sin(2*pi*t*wDrive)
    tint = tint - pi4Time
    
    if tint < ramseyDelay:
        return 0.0
    tint = tint - ramseyDelay
    
    if tint < pi4Time:
        return 0.1*sin(2*pi*t*wDrive)
    tint = tint - pi4Time
    return 0


def qubit_eval(externalDriveLevels):
    H = hplank * omega  * (Aplus*Aminus) + 0.5*hplank*epsilon*SigmaZ + hplank*g*eField*SigmaX
    Ht = [e* Cx *(Cq + Cg) / (Cq * Cg * 4.125e-6) * SigmaX, externalDriveLevels]

    gamma = omega*0.69/Q
    temp = 0.0 # Temperature, [GHz]

    cOps = [Aminus*sqrt(gamma*(1+temp/omega)), 
            Aplus*sqrt(gamma*temp/omega),
            SigmaP*sqrt(temp/epsilon/T1),
            SigmaN*sqrt((1+temp/epsilon)/T1)]

    result = mesolve([H, Ht], tensor(resonatorStartState, qubitStartState), 
        times, cOps, 
        [eField, Aplus*Aminus, SigmaZ, SigmaX, SigmaPopulation],
        options=Options(store_states=True))
    # test

    end_dm = result.states[-1]
    end_dm = end_dm.ptrace(1)
    loss = 1. - fidelity(end_dm, qubitTgtState)
    return result, loss


def report(result, picture = None):
    pp.subplot(2, 3, 1)
    pp.plot(result.times, result.expect[2])
    pp.title('SigmaZ')
    pp.subplot(2, 3, 3)
    pp.plot(result.times, result.expect[1])
    pp.title('Resonator occupancy')
    pp.subplot(2, 3, 2)
    pp.plot(result.times, result.expect[3])
    pp.title('SigmaX')
    pp.subplot(2, 3, 4)
    pp.plot(result.times, result.expect[0])
    pp.title('Resonator field')
    pp.subplot(2, 3, 6)
    if picture is not None:
        pp.savefig(picture)
    else:
        pp.show()


def main():
    pic = "dummy.png"
    externalDriveLevels = np.array([
        dummyExternalDrive(t, qubitStartState, qubitTgtState) for t in times])
    result, loss = qubit_eval(externalDriveLevels)
    report(result, picture=pic)
    print "Loss:", loss, "\npicture:", pic


if __name__ == '__main__':
    main()
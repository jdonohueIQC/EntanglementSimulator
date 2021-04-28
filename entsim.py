# Library for the entanglement simulator. Don't peek too much if you don't want spoilers.

from scipy import *
from numpy import *
from time import sleep
from qutip import Bloch, basis, Bloch3d
from ipywidgets import interact, IntSlider, FloatSlider, fixed
# NOTE: SymPy is not compatible with array commands


vec0 = array([[1],[0]])
vec1 = array([[0],[1]])
vecPlus = (vec0+vec1)/sqrt(2)
vecMinus = (vec0-vec1)/sqrt(2)
vecRight = (vec0+1j*vec1)/sqrt(2)
vecLeft = (vec0-1j*vec1)/sqrt(2)

vecH = vec0
vecV = vec1
vecD = vecPlus
vecA = vecMinus
vecR = vecRight
vecL = vecLeft

def GenState(theta=0,phi=0):
    return cos(theta/2)*vec0+sin(theta/2)*exp(1j*phi)*vec1
    
def CT(matrix): # quick conjugate transpose
    return conj(transpose(matrix))

def Proj(vec): # convert vectors to matrices
    return matmul(vec,CT(vec))

Id = array([[1,0],[0,1]])/2

# Bell states
PhiPlus = (kron(vec0,vec0)+kron(vec1,vec1))/sqrt(2)
PhiMinus = (kron(vec0,vec0)-kron(vec1,vec1))/sqrt(2)
PsiPlus = (kron(vec0,vec1)+kron(vec1,vec0))/sqrt(2)
PsiMinus = (kron(vec0,vec1)-kron(vec1,vec0))/sqrt(2)
PsiPlusImag = (kron(vec0,vec1)+1j*kron(vec1,vec0))/sqrt(2)
PhiPlusImag = (kron(vec0,vec0)+1j*kron(vec1,vec1))/sqrt(2)

# Test states for one-qubit tomography
OneStateA = Proj(GenState(1.52,0.02))*0.93+Id*0.07
OneStateB = Proj(GenState(1.01,pi/2))*0.5 + Id*0.5
OneStateC = Proj(GenState(arccos(1/sqrt(3)),pi/4))*0.99 + Id*0.01
OneStates = array([OneStateA,OneStateB,OneStateC])

# Test states for two-qubit tomography
TwoStateA = Proj(PsiMinus)*0.95+Proj(kron(vec0,vec0))*0.02+Proj(kron(vec1,vec1))*0.02+kron(Id,Id)*0.01
TwoStateB = Proj(PhiPlusImag)*0.68+Proj(kron(vec0,vec1))*0.3+kron(Id,Id)*0.02
TwoStateC = kron(Proj(GenState(pi/2,pi/3))*0.98+0.02*Id,Proj(GenState(pi/3,pi/12))*0.9+0.1*Id)
TwoStates = array([TwoStateA,TwoStateB,TwoStateC])

# note: Kronecker product default as kron(mat1,mat2) in numpy

def rot(angle): # generic counter-clockwise rotation matrix, to be used in the form rot(x).mat.rot(-x)
    return array([[cos(angle),-sin(angle)],[sin(angle),cos(angle)]])

def HWP(angle): # Half-wave plate rotation matrix as a function of the waveplate angle
    return matmul(matmul(rot(angle),array([[1,0],[0,-1]])),rot(-1*angle))

def QWP(angle): # Quarter-wave plate rotation matrix as a function of the waveplate angle
    return matmul(matmul(rot(angle),array([[1,0],[0,-1j]])),rot(-1*angle))#*exp(pi/4*1j)

def GenWaveplate(angle,biref): # General waveplate. Biref = pi for HWP
    return matmul(matmul(rot(angle),array([[1,0],[0,exp(1j*biref)]])),rot(-1*angle))#*exp(pi/4*1j)

def Pol(angle): # Polarizer (angle=0 gives a horizontal polarizer)
    return matmul(matmul(rot(angle),array([[1,0],[0,0]])),rot(-1*angle))#*exp(pi/4*1j)

def fwhm2sigma(fwhm):
    return fwhm/(2*sqrt(2*log(2)))

def PoissSamp(mean):
    return random.poisson(mean,1)[0]

#Convert a NumPy state to a QuTip state for Bloch sphere plotting
def VecToQutip(state):
    return state[0,0]*basis(2,0)+state[1,0]*basis(2,1)

def StateToBloch(state):
    rho = Proj(state)
    return array([2*real(rho[0,1]),2*imag(rho[1,0]),real(rho[0,0]-rho[1,1])])

def PlotBlochState(state):
    bsph = Bloch()
    bsph.add_vectors(StateToBloch(state))
    return bsph.show()

def PlotStateHWP(InputState,hwpAngle):
    OutputState=HWP(hwpAngle)@InputState
    pntsX = [StateToBloch(HWP(th)@InputState)[0] for th in arange(0,pi,pi/64)]
    pntsY = [StateToBloch(HWP(th)@InputState)[1] for th in arange(0,pi,pi/64)]
    pntsZ = [StateToBloch(HWP(th)@InputState)[2] for th in arange(0,pi,pi/64)]
    pnts = [pntsX,pntsY,pntsZ]
    bsph = Bloch()
    bsph.add_points(pnts)
    bsph.add_vectors(StateToBloch(OutputState))
    bsph.add_vectors(StateToBloch(InputState))
    return bsph.show()

def PlotStateQWP(InputState,qwpAngle):
    OutputState=QWP(qwpAngle)@InputState
    pntsX = [StateToBloch(QWP(th)@InputState)[0] for th in arange(0,pi,pi/64)]
    pntsY = [StateToBloch(QWP(th)@InputState)[1] for th in arange(0,pi,pi/64)]
    pntsZ = [StateToBloch(QWP(th)@InputState)[2] for th in arange(0,pi,pi/64)]
    pnts = [pntsX,pntsY,pntsZ]
    bsph = Bloch()
    bsph.add_points(pnts)
    bsph.add_vectors(StateToBloch(OutputState))
    bsph.add_vectors(StateToBloch(InputState))
    return bsph.show()

def PlotStateAnalyzer(InputState,hwpAngle,qwpAngle):
    QWPstate = QWP(qwpAngle)@InputState
    OutputState=HWP(hwpAngle)@QWP(qwpAngle)@InputState
    pntsX = [StateToBloch(QWP(th)@InputState)[0] for th in arange(0,pi,pi/64)]
    pntsY = [StateToBloch(QWP(th)@InputState)[1] for th in arange(0,pi,pi/64)]
    pntsZ = [StateToBloch(QWP(th)@InputState)[2] for th in arange(0,pi,pi/64)]
    pnts = [pntsX,pntsY,pntsZ]
    pntsX2 = [StateToBloch(HWP(th)@QWP(qwpAngle)@InputState)[0] for th in arange(0,pi,pi/64)]
    pntsY2 = [StateToBloch(HWP(th)@QWP(qwpAngle)@InputState)[1] for th in arange(0,pi,pi/64)]
    pntsZ2 = [StateToBloch(HWP(th)@QWP(qwpAngle)@InputState)[2] for th in arange(0,pi,pi/64)]
    pnts2 = [pntsX2,pntsY2,pntsZ2]
    bsph = Bloch()
    bsph.add_points(pnts)
    bsph.add_points(pnts2)
    bsph.add_vectors(StateToBloch(OutputState))
    bsph.add_vectors(StateToBloch(InputState))
    bsph.add_vectors(StateToBloch(QWPstate))
    return bsph.show()

def BlochHWP(InitState):
    return interact(PlotStateHWP, InputState=fixed(InitState), hwpAngle = FloatSlider(min=-pi,max=pi,value=0,step=pi/32))

def BlochQWP(InitState):
    return interact(PlotStateQWP, InputState=fixed(InitState), qwpAngle = FloatSlider(min=-pi,max=pi,value=0,step=pi/32))

def AnalyzerBloch(InitState):
    return interact(PlotStateAnalyzer,InputState=fixed(InitState), hwpAngle = FloatSlider(min=-pi,max=pi,value=0,step=pi/32), qwpAngle = FloatSlider(min=-pi,max=pi,value=0,step=pi/32))

SourceRate = 10000 # base count rate for the two-photon source

def OneQBmeasTest(hwpT=0,qwpT=0,meastime=1,stateVec=vecH): # For testing setup with known states
    print("Connected to source:\n", stateVec)
    print("Setting wave plates...") 
    sleep(0.25)
    PhotState = Proj(stateVec)*0.99 + Id*0.01 # Add some identity element to prevent things from being too deterministic / simulate effect of dark counts
    print("Wave plates set...")
    print("Measuring counts...")
    sleep(meastime)
    wps = HWP(hwpT)@QWP(qwpT)
    PlusRate = abs(trace(Proj(vec0)@wps@PhotState@CT(wps))) # the rate for the plus detector, absolute value to remove the (hopefully zero) complex part
    PlusCounts = PoissSamp(SourceRate*PlusRate*meastime)
    MinusRate = abs(trace(Proj(vec1)@wps@PhotState@CT(wps)))
    MinusCounts = PoissSamp(SourceRate*MinusRate*meastime)
    #return print(PlusRate, "Plus detector counts: ", PlusCounts, "\n", MinusRate, "Minus detector counts: ", MinusCounts)
    print("Plus detector counts: ", PlusCounts, "\nMinus detector counts: ", MinusCounts)
    return array([PlusCounts,MinusCounts])

def OneQBmeas(hwpT=0,qwpT=0,meastime=1,state=0):
    print("Connected to source ", state)
    print("Setting wave plates...") 
    sleep(0.25)
    print("Wave plates set...")
    print("Measuring counts...")
    sleep(meastime)
    wps = HWP(hwpT)@QWP(qwpT)
    PlusRate = abs(trace(Proj(vec0)@wps@OneStates[state]@CT(wps))) # the rate for the plus detector, absolute value to remove the (hopefully zero) complex part
    PlusCounts = PoissSamp(SourceRate*PlusRate*meastime)
    MinusRate = abs(trace(Proj(vec1)@wps@OneStates[state]@CT(wps)))
    MinusCounts = PoissSamp(SourceRate*MinusRate*meastime)
    #return print(PlusRate, "Plus detector counts: ", PlusCounts, "\n", MinusRate, "Minus detector counts: ", MinusCounts)
    print("Plus detector counts: ", PlusCounts, "\nMinus detector counts: ", MinusCounts)
    return array([PlusCounts,MinusCounts])

def TwoQBmeasBell(hwpT1=0,hwpT2=0,meastime=1,state=0):
    print("Connected to source", state)
    print("Setting wave plates...")
    sleep(0.25)
    print("Wave plates set...")
    print("Measuring counts...")
    sleep(meastime)
    wps1 = HWP(hwpT1)
    wps2 = HWP(hwpT2)
    wps = kron(wps1,wps2)
    # all the rates, including singles
    Plus1 = PoissSamp(SourceRate*meastime* abs(trace( kron(Proj(vec0),2*Id) @ wps @ TwoStates[state] @ CT(wps) )) )
    Plus2 = PoissSamp(SourceRate*meastime* abs(trace( kron(2*Id,Proj(vec0)) @ wps @ TwoStates[state] @ CT(wps) )) )
    Minus1 = PoissSamp(SourceRate*meastime* abs(trace( kron(Proj(vec1),2*Id) @ wps @ TwoStates[state] @ CT(wps) )) )
    Minus2 = PoissSamp(SourceRate*meastime* abs(trace( kron(2*Id,Proj(vec1)) @ wps @ TwoStates[state] @ CT(wps) )) )
    
    # For coincidences, assume a Klyshko efficiency of approximately 0.13 in each arm.
    
    PlusPlus = PoissSamp(0.13*SourceRate*meastime* abs(trace( Proj(kron(vec0,vec0)) @ wps @ TwoStates[state] @ CT(wps) )) )
    PlusMinus = PoissSamp(0.13*SourceRate*meastime* abs(trace( Proj(kron(vec0,vec1)) @ wps @ TwoStates[state] @ CT(wps) )) )
    MinusPlus = PoissSamp(0.13*SourceRate*meastime* abs(trace( Proj(kron(vec1,vec0)) @ wps @ TwoStates[state] @ CT(wps) )) )
    MinusMinus = PoissSamp(0.13*SourceRate*meastime* abs(trace( Proj(kron(vec1,vec1)) @ wps @ TwoStates[state] @ CT(wps) )) )
    
    print("\nSingles Counts 1:\nPlus: ", Plus1, "\nMinus: ", Minus1, "\nSingles Counts 2:\nPlus: ", Plus2, "\nMinus: ", Minus2, "\n\nCoincidence Counts:\n++ ", PlusPlus, "\n+- ", PlusMinus, "\n-+ ", MinusPlus, "\n-- ", MinusMinus)
    
    return array([Plus1,Minus1,Plus2,Minus2,PlusPlus,PlusMinus,MinusPlus,MinusMinus])


def TwoQBmeas(hwpT1=0,qwpT1=0,hwpT2=0,qwpT2=0,meastime=1,state=0):
    print("Connected to source", state)
    print("Setting wave plates...")
    sleep(0.25)
    print("Wave plates set...")
    print("Measuring counts...")
    sleep(meastime)
    wps1 = HWP(hwpT1)@QWP(qwpT1)
    wps2 = HWP(hwpT2)@QWP(qwpT2)
    wps = kron(wps1,wps2)
    # all the rates, including singles
    Plus1 = PoissSamp(SourceRate*meastime* abs(trace( kron(Proj(vec0),2*Id) @ wps @ TwoStates[state] @ CT(wps) )) )
    Plus2 = PoissSamp(SourceRate*meastime* abs(trace( kron(2*Id,Proj(vec0)) @ wps @ TwoStates[state] @ CT(wps) )) )
    Minus1 = PoissSamp(SourceRate*meastime* abs(trace( kron(Proj(vec1),2*Id) @ wps @ TwoStates[state] @ CT(wps) )) )
    Minus2 = PoissSamp(SourceRate*meastime* abs(trace( kron(2*Id,Proj(vec1)) @ wps @ TwoStates[state] @ CT(wps) )) )
    
    # For coincidences, assume a Klyshko efficiency of approximately 0.13 in each arm.
    
    PlusPlus = PoissSamp(0.13*SourceRate*meastime* abs(trace( Proj(kron(vec0,vec0)) @ wps @ TwoStates[state] @ CT(wps) )) )
    PlusMinus = PoissSamp(0.13*SourceRate*meastime* abs(trace( Proj(kron(vec0,vec1)) @ wps @ TwoStates[state] @ CT(wps) )) )
    MinusPlus = PoissSamp(0.13*SourceRate*meastime* abs(trace( Proj(kron(vec1,vec0)) @ wps @ TwoStates[state] @ CT(wps) )) )
    MinusMinus = PoissSamp(0.13*SourceRate*meastime* abs(trace( Proj(kron(vec1,vec1)) @ wps @ TwoStates[state] @ CT(wps) )) )
    
    print("\nSingles Counts 1:\nPlus: ", Plus1, "\nMinus: ", Minus1, "\nSingles Counts 2:\nPlus: ", Plus2, "\nMinus: ", Minus2, "\n\nCoincidence Counts:\n++ ", PlusPlus, "\n+- ", PlusMinus, "\n-+ ", MinusPlus, "\n-- ", MinusMinus)
    
    return array([Plus1,Minus1,Plus2,Minus2,PlusPlus,PlusMinus,MinusPlus,MinusMinus])
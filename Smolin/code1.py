import pennylane as qml
import numpy as np
import torch
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import torch.nn as nn
import math
from tqdm import trange
from itertools import product
import os

name = 'data3'

if not os.path.exists('./{}'.format(name)):
    os.mkdir('./{}'.format(name))

#---------------------------------------------------------------------------------------------------------

def Smolin(p):
	
    psi1 = np.array([[1/np.sqrt(2)],[0],[0],[1/np.sqrt(2)]])
    psi2 = np.array([[1/np.sqrt(2)],[0],[0],[-1/np.sqrt(2)]])
    psi3 = np.array([[0],[1/np.sqrt(2)],[1/np.sqrt(2)],[0]])
    psi4 = np.array([[0],[1/np.sqrt(2)],[-1/np.sqrt(2)],[0]])

    rho1 = np.matmul( psi1, psi1.T )
    rho2 = np.matmul( psi2, psi2.T )
    rho3 = np.matmul( psi3, psi3.T )
    rho4 = np.matmul( psi4, psi4.T )

    rho = (np.kron(rho1,rho1)+np.kron(rho2,rho2)+np.kron(rho3,rho3)+np.kron(rho4,rho4))/4
    return torch.from_numpy((1-p)*rho + (p/16)*np.eye(16)).to(torch.complex128)

#---------------------------------------------------------------------------------------------------------

def qlayer1(n,N):
	
    n1 = 2*n
    n2 = n+N
    NN = 0
    if n1 >= n2:
        NN = n1
    else:
        NN = n2

    dev = qml.device('default.qubit',wires= NN )

    @qml.qnode(dev,interface='torch')
    def f(rho):

        #rho
        wr, vr = eigh(rho)

        qml.AmplitudeEmbedding(features=np.sqrt(wr), wires= list(range(n)) )
        for i in range(n):
            qml.CNOT(wires=[i,n+i])

        qml.QubitUnitary(vr, wires= list(range(n)) )

        return qml.state()
    return f

#---------------------------------------------------------------------------------------------------------


def MultiControlU(param,wires=None):
    valores = [0, 1]  # Dois valores possíveis
    combinacoes = list(product(valores, repeat=len(wires)-1))
    for i in range(len(combinacoes)):
        param_ = param[i*3:i*3+3]
        a = param_[0]*math.pi*4
        b = param_[1]*math.pi*4
        c = param_[2]*math.pi*4
        qml.ctrl(qml.Rot, control=wires[:-1],control_values=combinacoes[i])(a,b,c, wires=wires[-1])



def rhoV(n,N,l1,l2):
    n1 = 2*n
    n2 = n+N
    NN = 0
    if n1 >= n2:
        NN = n1
    else:
        NN = n2

    dev = qml.device('default.qubit',wires= NN )

    @qml.qnode(dev,interface='torch')
    def f(param1,param2,param3,param4):

        #rho_qc
        for i in range(l1):
            for j in range(N):
                qml.RY(param1[j][i]*math.pi,wires=n+j)
            for j in range(N-1):
                qml.CNOT(wires=[n+j,n+j+1])

        for i in range(n):
            aa = list(range(n,n+N))
            aa.append( n-i -1)

            MultiControlU(param2[i],wires=aa)

        #U
        pi2 = math.pi*4
        for i in range(l2):
            for j in range(N):
                qml.Rot(param3[j][i][0]*pi2,param3[j][i][1]*pi2,param3[j][i][2]*pi2,wires=n+j)

            m = 0
            for j in range(N):
                for k in range(N):
                    if j!=k:
                        qml.CRot(param4[m][0]*pi2,param4[m][1]*pi2,param4[m][1]*pi2,wires=[n+j,n+k])
                        m+=1


        return qml.state()
    return f



class Model(nn.Module):
    def __init__(self,rho,n,N,l1,l2):
        super(Model, self).__init__()

        #torch.manual_seed(42)
        self.param1 = nn.Parameter(torch.rand(N,l1))
        self.param2 = nn.Parameter(torch.rand(n,3*(2**N)))
        self.param3 = nn.Parameter(torch.rand(N,l2,3))
        self.param4 = nn.Parameter(torch.rand(N*(N-1)*l2,3))

        self.f = nn.Sigmoid()

        self.ql2 = rhoV(n,N,l1,l2)
        self.state1 = qlayer1(n,N)(rho)
        self.state1 = self.state1.view(self.state1.size(0),-1)


    def forward(self):

        y1 = self.f(self.param1)
        y2 = self.f(self.param2)
        y3 = self.f(self.param3)
        y4 = self.f(self.param4)
        state2 = self.ql2(y1,y2,y3,y4)
        state2 = state2.view(state2.size(0),-1)

        fid = torch.abs(torch.matmul( torch.conj(self.state1).T , state2 ))
        dd = torch.abs(1-fid)
        return dd

#-----------------------------------------------------------------------

n = 4
N = 5
l1 = 2
l2 = 4*(N+n)



epochs = 3000
lr = 0.01

p = [0.6]


for k in range(len(p)):
    print('\n --------------------- p{} --------------------- \n'.format(p[k]))
    if not os.path.exists('./{}/em_p_{}'.format(name,p[k])):
        os.mkdir('./{}/em_p_{}'.format(name,p[k]))

    rho = Smolin(p[k])
    for kk in range(10):
        net = Model(rho,n,N,l1,l2)

        optim = torch.optim.Adam(net.parameters(),lr=lr)

        dataHist = []
        for i in trange(epochs):
            out1 = net()
            dataHist.append( out1.item() )
            optim.zero_grad()
            out1.backward()
            optim.step()

        np.savetxt('./{}/em_p_{}/model_{}.txt'.format(name,p[k],kk),dataHist)

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


#---------------------------------------------------------------------------------------------------------

def Werner(eta):
    psi = np.array([[1/np.sqrt(2)],[0],[0],[1/np.sqrt(2)]])
    return torch.from_numpy(((1-eta)/4)*np.eye(4) + eta*np.matmul( psi, psi.T )).to(torch.complex128)


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



def rhoV(n,N,l1):
    n1 = 2*n
    n2 = n+N
    NN = 0
    if n1 >= n2:
        NN = n1
    else:
        NN = n2

    dev = qml.device('default.qubit',wires= NN )

    @qml.qnode(dev,interface='torch')
    def f(param1,param2,param3):

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
        qml.ArbitraryUnitary(param3, wires=range(n,N+n))
        


        return qml.state()
    return f





class Model(nn.Module):
    def __init__(self,rho,n,N,l1):
        super(Model, self).__init__()

        #torch.manual_seed(42)
        self.param1 = nn.Parameter(torch.rand(N,l1))
        self.param2 = nn.Parameter(torch.rand(n,3*(2**N)))
        self.param3 = nn.Parameter(torch.rand(4**N-1))
        
        

        self.ql2 = rhoV(n,N,l1)
        self.state1 = qlayer1(n,N)(rho)
        self.state1 = self.state1.view(self.state1.size(0),-1)


    def forward(self):

       
        state2 = self.ql2(self.param1,self.param2,self.param3)
        state2 = state2.view(state2.size(0),-1)

        fid = torch.abs(torch.matmul( torch.conj(self.state1).T , state2 ))
        dd = torch.abs(1-fid)
        return dd

#-----------------------------------------------------------------------

name = 'data1'

if not os.path.exists('./{}'.format(name)):
    os.mkdir('./{}'.format(name))


n = 2
N = 2
l1 = 1



epochs = 200
lr = 0.01

x1 = np.arange(0,0.96,0.05)
x2 = np.arange(0.96,1,0.01)

p = np.concatenate([x1,x2])

for k in range(len(p)):
    print('\n --------------------- p{} --------------------- \n'.format(p[k]))
    if not os.path.exists('./{}/em_p_{}'.format(name,p[k])):
        os.mkdir('./{}/em_p_{}'.format(name,p[k]))

    rho = Werner(p[k])
    for kk in range(5):
        net = Model(rho,n,N,l1)

        optim = torch.optim.Adam(net.parameters(),lr=lr)

        dataHist = []
        for i in trange(epochs):
            out1 = net()
            dataHist.append( out1.item() )
            optim.zero_grad()
            out1.backward()
            optim.step()

        np.savetxt('./{}/em_p_{}/model_{}.txt'.format(name,p[k],kk),dataHist)

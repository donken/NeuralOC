# Cross2D.py
# problem class for 2-D agents

import torch
import math
from src.Phi import *
from src.utils import normpdf


class Cross2D:
    """
    attributes:
        xtarget - target state
        d       - number of dimensions
        nAgents - number of agents
        alph_Q  - obstacle height / weight multiplier for Q
        alph_W  - weight multiplier for interaction costs

    methods:
        train()          - set to training mode
        eval()           - set to evaluation mode
        calcGradpH(x, p) - calculate gradient of Hamiltonian wrt p
        calcLHQW         - calculate the Lagrangian and Hamiltonian
        calcQ            - calculate the obstacle/terrain cost Q
        calcObstacle     - calculate the obstacle/terrain costs for single agent Q_i
        calcW            - calculate the interaction costs W
        calcCtrls        - calculate the controls
    """

    def __init__(self, xtarget, obstacle=None, alph_Q=1.0, alph_W=1.0, r=0.5):
        self.xtarget = xtarget.squeeze() # G assumes the traget is squeezed
        self.d = xtarget.numel()
        self.agentDim = 2
        self.obstacle = obstacle
        self.alph_Q = alph_Q
        self.alph_W = alph_W

        # agents
        self.nAgents = self.d // self.agentDim
        self.r = r
        self.training = True # obstacle settings and self.r vary between training and validation

        if self.obstacle == 'softcorridor':
            self.mu1 = torch.tensor([-2.5, 0.], dtype=xtarget.dtype, device=xtarget.device).view(1, -1)
            self.mu2 = torch.tensor([ 2.5, 0.], dtype=xtarget.dtype, device=xtarget.device).view(1, -1)
            self.mu3 = torch.tensor([-1.5, 0.], dtype=xtarget.dtype, device=xtarget.device).view(1, -1)
            self.mu4 = torch.tensor([ 1.5, 0.], dtype=xtarget.dtype, device=xtarget.device).view(1, -1)
            self.cov = torch.tensor([ 0.2, 0.2],dtype=xtarget.dtype, device=xtarget.device).view(1, -1)
        elif self.obstacle == 'hardcorridor':
            self.mu1 = torch.tensor([0., 4.], dtype=xtarget.dtype, device=xtarget.device).view(1,-1)
            self.mu2 = torch.tensor([0., -3.5], dtype=xtarget.dtype, device=xtarget.device).view(1,-1)
            self.cov = torch.tensor([1., 1.], dtype=xtarget.dtype, device=xtarget.device).view(1,-1)

    def __repr__(self):
        return "Cross2D Object"

    def __str__(self):
        s = "Cross2D Object \n d = {:} \n nAgents = {:} \n xtarget = {:} \n obstacle:{:}".format(
            self.d, self.nAgents, self.xtarget, self.obstacle
        )
        return s

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def calcGradpH(self, x, p):
        return p


    def calcLHQW(self, x, p):

        Q = self.alph_Q * self.calcQ(x)

        L = 0.5 * torch.sum( p**2, dim=1, keepdims=True) + Q

        if self.alph_W != 0.0:
            W = self.calcW(x, p)
            L = L + self.alph_W * W
        else:
            W = 0.0*L # zeros as placeholders

        H = -L + torch.sum( p**2, dim=1, keepdims=True)

        return L, H, Q, W


    def calcObstacle(self, xAgent):

        if self.obstacle == 'softcorridor':
            Q1 = normpdf(xAgent, mu=self.mu1, cov=self.cov)
            Q2 = normpdf(xAgent, mu=self.mu2, cov=self.cov)
            Q3 = normpdf(xAgent, mu=self.mu3, cov=self.cov)
            Q4 = normpdf(xAgent, mu=self.mu4, cov=self.cov)
            Q = Q1 + Q2 + Q3 + Q4
            return Q
        if self.obstacle == 'hardcorridor':
            Q1 = normpdf(xAgent, mu=self.mu1, cov=self.cov)
            Q2 = normpdf(xAgent, mu=self.mu2, cov=self.cov)
            Q = Q1 + Q2

            # if want to make hard constraint
            # put gaussians inside a box
            # the obstacle is smaller in validation than in training... varies with the radius r
            # we want hard circular obstacles with radius 2
            if self.training:
                mask = ( torch.norm(xAgent - self.mu1,dim=1) < 2.0 + self.r) | \
                       ( torch.norm(xAgent - self.mu2,dim=1) < 2.0 + self.r)
            else:
                mask = (torch.norm(xAgent - self.mu1,dim=1) < 2.0 ) | (torch.norm(xAgent - self.mu2,dim=1) < 2.0)
                return mask

            Q[~mask] = 0.0
            return Q

        else:
            return 0.0*xAgent

    def calcQ(self, x):

        if self.obstacle is not None:
            shp = x.shape
            q = self.calcObstacle(x.reshape(-1,self.agentDim))
            return torch.sum(q.reshape(shp[0],-1), dim=1, keepdim=True)
        else:
            return 0.0 * x[:,0].unsqueeze(1)

    def calcW(self, x, p):

        if self.nAgents == 1:
            return 0.0 * x[:, 0]
        elif self.nAgents==2:

            # 2-norm
            dists = torch.norm(x[:, 0:2] - x[:, 2:4], p=2, dim=1, keepdim=True)

            if self.training: # train with a 10% larger r
                mask = dists <  2.2 * self.r
            else:
                mask = dists < 2 * self.r

            W = torch.exp( - dists**2 / (2*self.r**2) )
            return mask * W

        elif self.nAgents > 2:
            x = x.view(x.size(0), self.nAgents, 2)

            dists = torch.norm( x.reshape(x.size(0), self.nAgents,1, 2 )  - x.reshape(x.size(0), 1, self.nAgents, 2 ), p=2, dim=3 )

            if self.training: # train with a 10% larger r
                mask = dists <  2.2 * self.r
            else:
                mask = dists < 2 * self.r
            dists = torch.exp( - (mask*dists)**2 / (2*self.r**2))
            mask2 = dists == 1.
            res = (   (dists.sum(dim=[1,2]) - mask2.sum(dim=[1,2])) / 2. ).view(-1,1) # divide by 2 bc counting distances twice

            return res

        return 0.0 * x[:, 0]

    def calcCtrls(self, x , p):
        return -p


if __name__ == '__main__':

    # test the Cross2D Object
    d = 2
    init = -1.5
    nex = 20
    x0 = torch.Tensor([init, init]) + torch.randn(nex, 2)
    xtarget = torch.Tensor([2, 2]).unsqueeze(0)

    prob = Cross2D(xtarget)

    p = torch.randn(x0.shape)
    newL, newH, newQ, newW = prob.calcLHQW(x0,p)
    newGradpH  = prob.calcGradpH(x0,p)

    print("L norm: ", torch.norm(newL).item())
    print("H norm: ", torch.norm(newH).item())
    print("grad_p(H) norm: ", torch.norm(newGradpH).item())


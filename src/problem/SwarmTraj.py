# SwarmTraj.py
# inspired by the Honig et al. 2018 paper Trajectory Planning for Quadrotor Swarms
# agent is just a 3-dimensional object
import torch
from torch.nn.functional import pad
from src.Phi import *
from src.utils import normpdf
import math



class SwarmTraj:
    """
    attributes:
        xtarget  - target state
        d        - number of dimensions
        nAgents  - number of agents
        agentDim - dimensionality of each agent
        alph_Q   - alpha weight on the obstacle cost
        alph_W   - alpha wieght on the interactions cost
        obstacle - string, name of the obstacle setup
        r        - float, radius of an agent
        training - boolean, if set to True during training and False during validation

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
        self.xtarget = xtarget.squeeze() # G assumes the target is squeezed
        self.d = xtarget.numel()
        self.agentDim = 3
        self.obstacle = obstacle
        self.alph_Q = alph_Q
        self.alph_W = alph_W
        self.nAgents = self.d // self.agentDim
        self.r = r
        self.training = True # so obstacle settings and self.r vary between training and validation

        if self.obstacle =='blocks':
            self.mu1 = torch.tensor([ 0. , 0. , 2.], dtype=xtarget.dtype, device=xtarget.device).view(1, -1)
            self.mu2 = torch.tensor([ 2.5 , 0. , 2.], dtype=xtarget.dtype, device=xtarget.device).view(1, -1)
            self.cov1 = 3. * torch.tensor([ 3., 1. , 3.], dtype=xtarget.dtype, device=xtarget.device).view(1, -1)
            self.cov2 = 3. * torch.tensor([ 3., 1. , 1.], dtype=xtarget.dtype, device=xtarget.device).view(1, -1)

    def __repr__(self):
        return "SwarmTraj Object"

    def __str__(self):
        s = "SwarmTraj Object \n d = {:} \n nAgents = {:} \n xtarget = {:} \n obstacle:{:}".format(
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

        if self.alph_Q > 0:
            Q = self.calcQ(x).view(-1,1)
        else:
            Q = 0.*x[:,0].view(-1,1)

        L = 0.5 * torch.sum( p**2, dim=1, keepdims=True) + self.alph_Q * Q
        if self.alph_W != 0.0:
            W = self.calcW(x, p)
            L = L + self.alph_W * W
        else:
            W = 0.0*L # just for show

        H = -L + torch.sum( p**2, dim=1, keepdims=True)

        return L, H, Q, W


    def calcObstacle(self, xAgent):

        if self.obstacle == 'blocks':
            xpos = xAgent[:,0:3] # just deal with the quadcopter's x,y,z position

            Q1 = normpdf(xpos, mu=self.mu1, cov=self.cov1)
            Q2 = normpdf(xpos, mu=self.mu2, cov=self.cov2)
            Q  = Q1 + Q2  + 999.

            # put gaussians inside bounds.
            # blocks
            if self.training:
                mask = ( (xpos[:,0] < 2.0 + self.r) & (xpos[:,0] > -2.0 - self.r) &
                         (xpos[:,1] < 0.5 + self.r) & (xpos[:,1] > -0.5 - self.r) &
                         (xpos[:,2] < 7.0 + self.r) ) \
                     | ( (xpos[:, 0] < 4.0 + self.r) & (xpos[:, 0] > 2.0 - self.r) &
                         (xpos[:, 1] < 1.0 + self.r) & (xpos[:, 1] > -1.0 - self.r) &
                         (xpos[:, 2] < 4.0 + self.r))
            else:
                mask = ( (xpos[:,0] < 2.0 ) & (xpos[:,0] > -2.0 ) &
                         (xpos[:,1] < 0.5 ) & (xpos[:,1] > -0.5 ) &
                         (xpos[:,2] < 7.0 ) ) \
                     | ( (xpos[:, 0] < 4.0 ) & (xpos[:, 0] > 2.0 ) &
                         (xpos[:, 1] < 1.0 ) & (xpos[:, 1] > -1.0 ) &
                         (xpos[:, 2] < 4.0 ))

                return mask.unsqueeze(1)

            Q[~mask] = 0.0
            return Q

        else:
            return 0.0*xAgent

    def calcQ(self, x):

        if self.obstacle is not None:
            shp = x.shape
            q = self.calcObstacle(x.reshape(-1,self.agentDim)) # reshape and compute obstacle for each agent
            return torch.sum(q.reshape(shp[0],-1), dim=1, keepdim=True) # reshape back
        else:
            return 0.0 * x[:,0].unsqueeze(1)

    def calcW(self, x, p):

        if self.nAgents == 1:
            return 0.0 * x[:, 0]
        elif self.nAgents==2:

            dists = torch.norm(x[:, 0:3] - x[:, 3:6], p=2, dim=1, keepdim=True)
            if self.training:
                mask = dists < 2.2 * self.r
            else:
                mask = dists < 2 * self.r
            W = torch.exp( - dists**2 / (2*self.r**2) )
            return mask * W

        elif self.nAgents > 2:

            x = x.view(x.size(0), self.nAgents, 3)

            dists = torch.norm( x.reshape(x.size(0), self.nAgents,1, 3 )  - x.reshape(x.size(0), 1, self.nAgents, 3 ), p=2, dim=3 )
            
            if self.training:
                mask = dists < 3.2 * self.r
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

    # test the SwarmTraj Object

    nAgents = 32
    d = nAgents * 3
    nTrain=20
    xtarget = torch.tensor([-2., 2., 8.,
                                -1., 2., 8.,
                                0., 2., 8.,
                                1., 2., 8.,
                                2., 2., 8.,
                                -2.5, 3., 8.,
                                -1.5, 3., 8.,
                                -0.5, 3., 8.,
                                0.5, 3., 8.,
                                1.5, 3., 8.,
                                2.5, 3., 8.,
                                -2., 4., 8.,
                                -1., 4., 8.,
                                0., 4., 8.,
                                1., 4., 8.,
                                2., 4., 8.])

    xtarget = torch.cat((xtarget.view(-1, 3), torch.tensor([0, -0.5, -3]) + xtarget.view(-1, 3)), dim=0).view(-1)
    halfTrain = nTrain // 2
    xInit = torch.tensor([1, -1, -1]) * xtarget.view(-1, 3) + torch.tensor([0, 0, 10])
    xInit = xInit.view(1, -1)
    x0 = xInit + torch.randn(halfTrain, d)
    xmore = xtarget + torch.randn(halfTrain, d)
    x0 = torch.cat((x0, xmore), dim=0)
    for j in range(nAgents):
        x0[:, 3 * j + 3:3 * (j + 1)] = 0. * x0[:, 3 * j + 3:3 * (j + 1)]

    prob = SwarmTraj(xtarget, r=0.2)

    p = torch.randn(x0.shape)
    L,H,Q,W = prob.calcLHQW(x0,p)
    gradpH = prob.calcGradpH(x0,p)

    # calculate difference with standing approach
    print("L norm: ", torch.norm(L).item())
    print("H norm: ", torch.norm(H).item())
    print("grad_p(H) norm: ", torch.norm(gradpH).item())


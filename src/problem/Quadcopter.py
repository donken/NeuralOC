# Quadcopter.py
import torch
from torch.nn.functional import pad
from src.Phi import *
from src.utils import normpdf

class Quadcopter:
    """
    attributes:
        xtarget  - target state
        d        - number of dimensions
        nAgents  - number of agents
        agentDim - dimensionality of each agent
        mass     - mass of the quadcopter (all agents have same mass)
        grav     - gravity acceleration
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
        ---------
        calcU            - calculate thrust u
        f                - helper function
    """

    def __init__(self, xtarget, obstacle=None, alph_Q=1.0, alph_W=1.0, mass=1.0, grav=9.81, r = 1.0):
        self.xtarget = xtarget.squeeze() # G assumes the target is squeezed
        self.d = xtarget.numel()
        self.agentDim = 12
        self.nAgents = self.d // self.agentDim
        self.mass = mass
        self.grav = grav
        self.alph_W = alph_W
        self.obstacle = obstacle
        self.alph_Q = alph_Q
        self.r = r
        self.training = True # so obstacle settings and self.r vary during training and validation

    def __repr__(self):
        return "Quadcopter Object"

    def __str__(self):
        s = "Quadcopter Object Optimal Control \n d = {:} \n nAgents = {:} \n xtarget = {:} \n obstacle = {:}  \n alph_Q = {:}  \n mass = {:} \n grav = {:}".format(
            self.d, self.nAgents, self.xtarget, self.obstacle, self.alph_Q, self.mass, self.grav
        )
        return s

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def calcGradpH(self, xFull, pFull):

        grad_pH = torch.empty(0, device=xFull.device, dtype=xFull.dtype)

        for j in range(self.nAgents):
            # in  multi-agent, break into separate agents
            x = xFull[ : , 12*j : 12*(j+1) ]
            P = pFull[ : , 12*j : 12*(j+1)]
            u, f7c, f8c, f9c = self.calcU(x, P)

            # Xdot = grad_p H
            grad_pH = torch.cat((grad_pH,
                - x[:, 6:],
                - (u / self.mass) * f7c.view(-1,1),
                - (u / self.mass) * f8c.view(-1,1),
                - (u / self.mass) * f9c.view(-1,1) + self.grav,
                  (1. / 2.) * P[:, 9:12]  # this is P_k
            ), dim=1)

        return grad_pH

    def calcLHQW(self, xFull, pFull):

        H = 0.
        Q = self.calcQ(xFull).view(-1,1)
        L = self.alph_Q * Q

        if self.alph_W > 0.0:
            W = self.calcW(xFull, pFull).view(-1,1)
            L = L + self.alph_W * W
        else:
            W = 0.0 * L

        for i in range(self.nAgents):
            # in  multi-agent, break into separate agents
            x = xFull[ : , 12*i : 12*(i+1) ]
            P = pFull[ : , 12*i : 12*(i+1)]

            sumSqr =  (P[:, 9] ** 2 + P[:, 10] ** 2 + P[:, 11] ** 2).view(-1, 1)
            u , f7c, f8c, f9c = self.calcU(x, P)
            L = L +   2 + u ** 2 + 0.25 * sumSqr

            H = H  - L   \
                - torch.sum(x[:, 6:9]  * P[:, 0:3], dim=1, keepdims=True) \
                - torch.sum(x[:, 9:12] * P[:, 3:6], dim=1, keepdims=True) \
                - (u / self.mass) * (f7c * P[:, 6] + f8c * P[:, 7] + f9c * P[:, 8]).unsqueeze(1) \
                + self.grav*P[:,8].unsqueeze(1) + 0.5 * sumSqr

        return L, H, Q, W


    def calcObstacle(self, xAgent):

        if self.obstacle is None:
            return 0.0*xAgent
        else:
            print('obstacle ', self.obstacle, ' is not implemented')
            return 0.0 * xAgent

    def calcQ(self, x):

        if self.obstacle is not None:
            shp = x.shape
            q = self.calcObstacle(x.reshape(-1,self.agentDim))
            return torch.sum(q.reshape(shp[0],-1), dim=1, keepdim=True)
        else:
            return 0.0 * x[:,0].unsqueeze(1)


    def calcW(self, x, P):

        if self.nAgents==1:
            return (0.0 * x[:, 0]).view(-1,1)
        elif self.nAgents==2:
            dists = torch.norm(x[:, 0:3] - x[:, 12:15], p=2, dim=1, keepdim=True)
            mask = dists <  2*self.r
            W = torch.exp( - dists**2 / (2*self.r**2) )
            return mask * W

        elif self.nAgents > 2:
            x = x.view(x.size(0), self.nAgents, 12)
            x = x[0:3,:] # just the positional x,y,z

            dists = torch.norm( x.reshape(x.size(0), self.nAgents,1, 12 )  - x.reshape(x.size(0), 1, self.nAgents, 12 ), p=2, dim=3 )

            mask = dists < 2 * self.r
            dists = torch.exp( - (mask*dists)**2 / (2*self.r**2))
            mask2 = dists == 1.
            res = (   (dists.sum(dim=[1,2]) - mask2.sum(dim=[1,2])) / 2. ).view(-1,1) # divide by 2 bc counting distances twice

            return res


        return 0.0 * x[:, 0]

    def calcU(self, x, P):
        f7c, f8c, f9c = self.f(x[:, 3:6])
        u = -1 / (2 * self.mass) * (f7c * P[:, 6] + f8c * P[:, 7] + f9c * P[:, 8]).view(-1, 1)
        return u, f7c, f8c, f9c

    def calcCtrls(self, xFull , pFull):

        ctrls = torch.empty(0, device=xFull.device, dtype=xFull.dtype)
        for j in range(self.nAgents):
            x = xFull[ : , 12*j : 12*(j+1) ]
            p = pFull[ : , 12*j : 12*(j+1)]
            u, _, _, _ = self.calcU(x[:, 0:self.d], p)
            ctrls = torch.cat( (ctrls, u, -0.5 * p[:, 9:12] ) , dim=1) # concatenate the agent's controls to all of them

        return ctrls

    # ang: [ψ,θ,ϕ]
    # index 0 = ψ
    # index 1 = θ
    # index 2 = ϕ

    # combine all three
    def f(self, ang):
        sinPsi   = torch.sin(ang[:, 0])
        sinTheta = torch.sin(ang[:, 1])
        sinPhi   = torch.sin(ang[:, 2])
        cosPsi   = torch.cos(ang[:, 0])
        cosTheta = torch.cos(ang[:, 1])
        cosPhi   = torch.cos(ang[:, 2])

        # f7 = sin(ψ) sin(ϕ) + cos(ψ) sin(θ) cos(ϕ)
        f7   = sinPsi * sinPhi + cosPsi * sinTheta * cosPhi
        # f8 = - cos(ψ) sin(ϕ) + sin(ψ) sin(θ) cos(ϕ)
        f8   = - cosPsi * sinPhi + sinPsi * sinTheta * cosPhi
        # f9 = cos(θ) cos(ϕ)
        f9   = cosTheta * cosPhi

        return f7 , f8 , f9


if __name__ == '__main__':


    # test the Quadcopter Object
    d = 12
    init = -1.5
    nex = 20
    x0 = torch.Tensor([[init, init, init]]) + torch.randn(nex, 3)
    x0 = pad(x0, [0,d-3,0,0], value=0)
    xtarget = torch.Tensor([2, 2, 2]).unsqueeze(0)
    xtarget = pad(xtarget, [0, d - 3, 0, 0], value=0)

    prob = Quadcopter(xtarget)

    p = torch.randn(x0.shape)
    L,H,Q,W = prob.calcLHQW(x0,p)
    gradpH = prob.calcGradpH(x0,p)

    print("L norm: ", torch.norm(L).item())
    print("H norm: ", torch.norm(H).item())
    print("grad_p(H) norm: ", torch.norm(gradpH).item())


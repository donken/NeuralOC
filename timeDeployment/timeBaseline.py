# timeBaseline.py

import math
import torch
import argparse
from src.utils import normpdf
from src.initProb import *
from src.OCflow import ocG
from src.plotter import *
import time

parser = argparse.ArgumentParser('Baseline')
parser.add_argument(
    '--data', choices=['softcorridor','swap2','swap12','swarm','singlequad' ],
    type=str, default='softcorridor')

parser.add_argument("--nt"    , type=int, default=20, help="number of time steps")
parser.add_argument('--alph'  , type=str, default='10.0, 100.0, 10.0')
# alphas: G, Q (obstacle), W (interaction)
parser.add_argument('--niters', type=int, default=301)
parser.add_argument('--prec'  , type=str, default='single', choices=['single','double'], help="single or double precision")
parser.add_argument('--save'  , type=str, default='experiments/oc/baseline', help="define the save directory")
parser.add_argument('--resume'  , type=str, default=None, help="for loading a pretrained model")
parser.add_argument('--time_log', type=str, default='time_log', help='file to print timing the deployment/online mode')

args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

if args.prec =='double':
    argPrec = torch.float64
else:
    argPrec = torch.float32

device=  torch.device('cpu')
cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

nt = args.nt


# open time_log to print timing results to
timeFile = open(args.time_log, 'a')
print('\n----------\n', file = timeFile)
print('baseline', file = timeFile)
print('problem: ', args.data, file = timeFile)
print('args: ', args, file = timeFile)
print('device: ', device, file = timeFile)


def loss_fun(U, Z_0, prob, nt, alphG):
    """
    compute loss
    :param U:     nt-by-d tensor, set of controls
    :param Z_0:   d-dim vector, initial point
    :param prob:  problem Object
    :param nt:    int, number of time steps
    :param alphG: float, alpha_0 on the terminal cost G
    :return: float, loss value
    """
    h = 1. / nt
    Z = Z_0
    loss = 0
    for i in range(nt):
        Z = Z + h * U[i,:]
        L, _ , _ , _ = prob.calcLHQW(Z.view(1,-1), U[i,:].view(1,-1))
        loss = loss + h * L

    cG   = 0.5 * torch.sum(ocG(Z.view(1,-1), prob.xtarget)**2, 1, keepdims=True)
    loss = loss + alphG * cG
    return loss

def trainBaseline(z0,prob,nt=10, nIters = 600, alphG=100., u=None):
    """
    method to train the baseline model, a discrete optimization approach
    :param z0:      d-dim vector, initial point
    :param prob:    problem Object
    :param nt:      int, number of time steps
    :param nIters:  int, max number of iterations
    :param alphG:   float, alpha_0 on the terminal cost G
    :param u:       nt-by-d Parameters, the controls, initial guess
    :return: nt-by-d Parameters, the optimized u
    """

    timeForw = 0  # accumulate timing 
    timeBack = 0  # accumulate timing of gradient

    if u is None:
        # initialize with noisy straight lines
        y = prob.xtarget - z0
        u = y * torch.ones((nt,z0.numel()), device=y.device, dtype=y.dtype) + 0.1*torch.randn(nt, z0.numel(), device=y.device, dtype=y.dtype)
        u = torch.nn.Parameter(u)

    bestLoss = float("inf")
    ubest = torch.zeros_like(u.data)

    lr = 0.1
    optim = torch.optim.Adam([{'params': u}], lr=lr, weight_decay=0.0 )
    for i in range(nIters):

        optim.zero_grad()

        # -------time the computation of the Gradient-------
        start = time.time()
        err = loss_fun(u, z0, prob, nt, alphG)  # forward
        midTime = time.time()
        err.backward() # backprop
        end = time.time()

        if i > 1:  # the first one is slow startup, so drop it
            timeForw = timeForw + (midTime - start)
            timeBack = timeBack + (end - midTime)
            if i % 50 == 0:  # log_freq
                print(i, err.item())
                print('%5d   time: %5f  forw sec/itr: %5f  bk sec/itr: %5f  approx 100grads: %5f'
                      % (i, end - start, timeForw / i, timeBack / i, 100 * (timeForw / i + timeBack / i)),
                      file=timeFile)



        optim.step()

        err = loss_fun(u, z0, prob, nt, alphG) # calc loss
        if err.item() < bestLoss:
            bestLoss= err.item()
            ubest.data = copy.deepcopy(u.data)

        if nIters//4 == 0: # lower lr
            lr = lr * 0.1
            # print('lr: ',lr)

    return ubest


if __name__ == '__main__':

    alphG = args.alph[0]
    prob, _, _, xInit = initProb(args.data, 10, 10, var0=1.0, cvt=cvt,
                            alph=[alphG, args.alph[1], args.alph[2], 0.0, 0.0, 0.0])
    prob.train()
    d = xInit.numel()
    strTitle = 'baseline_' + args.data + '_{:}_{:}_{:}'.format(int(alphG), int(prob.alph_Q),int(prob.alph_W))

    x0 = xInit # x0 can be more than one point
    traj  = cvt(torch.zeros(x0.size(0), d, nt+1))
    h = 1. / nt
    for i in range(x0.size(0)):
        z0 = x0[i,:]


        uopt = trainBaseline(z0, prob, nt=nt, nIters = args.niters, alphG=alphG, u=None)

    timeFile.close()





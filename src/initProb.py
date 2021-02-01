# initProb.py
# initialize the OC problem

import torch
from src.problem.Quadcopter import *
from src.problem.SwarmTraj import *
from src.problem.Cross2D import *

def initProb(sData, nTrain, nVal, var0,  alph, cvt):
    """
    initialize the OC problem that we want to solve

    :param sData:  str, name of the problem
    :param nTrain: int, number of samples in a batch, drawn from rho_0
    :param nVal:   int, number of validation samples to draw from rho_0
    :param var0: float, variance of rho_0
    :param alph:  list, 6-value list of parameters/hyperparameters
    :param cvt:   func, conversion function for typing and device
    :return:
        prob:  the problem Object
        x0:    nTrain -by- d tensor, training batch
        x0v:   nVal -by- d tensor, training batch
        xInit: 1 -by- d tensor, center of rho_0
    """
    if sData == 'softcorridor':
        d = 4
        xtarget = cvt(torch.tensor([[2, 2, -2, 2]]))
        xInit   = cvt(torch.tensor([[-2, -2, 2, -2]]))
        x0      = xInit + cvt(var0 * torch.randn(nTrain, d))
        x0v     = xInit + cvt(var0 * torch.randn(nTrain, d))
        prob    = Cross2D(xtarget, obstacle='softcorridor', alph_Q=alph[1], alph_W=alph[2], r=0.5)

    elif sData == 'swarm':

        nAgents = 32
        d = nAgents*3
        xtarget = cvt(torch.tensor([-2., 2., 8.,
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
                                     2., 4., 8.]))

        xtarget = torch.cat((xtarget.view(-1, 3), cvt(torch.tensor([0, -0.5, -3])) + xtarget.view(-1, 3)), dim=0).view(-1)


        halfTrain = nTrain // 2

        xInit = cvt(torch.tensor([1,-1,-1])) * xtarget.view(-1,3) + cvt(torch.tensor([0,0,10]))
        xInit = xInit.view(1,-1)

        x0     = xInit  + cvt( var0 * torch.randn(halfTrain, d))
        xmore  = xtarget + cvt(var0 * torch.randn(halfTrain, d))
        x0 = torch.cat((x0, xmore), dim=0)


        # validation samples from rho_0
        x0v     = xInit  + cvt( var0 * torch.randn(halfTrain, d))
        for j in range(nAgents):
            x0[ :,3*j+3:3*(j+1)] = 0. *  x0[ :,3*j+3:3*(j+1)]
            x0v[:,3*j+3:3*(j+1)] = 0. *  x0v[:,3*j+3:3*(j+1)]

        prob = SwarmTraj(xtarget, obstacle='blocks', alph_Q=alph[1],  alph_W=alph[2], r= 0.2)


    elif sData == 'swarm50':

        nAgents = 50
        d = nAgents*3

        xtarget = cvt(torch.tensor([-2., 2., 6.,
                                    -1., 2., 6.,
                                     0., 2., 6.,
                                     1., 2., 6.,
                                     2., 2., 6.,
                                     3., 2., 6.,
                                     4., 2., 6.,
                                    -2.5, 3., 7.,
                                    -1.5, 3., 7.,
                                    -0.5, 3., 7.,
                                     0.5, 3., 7.,
                                     1.5, 3., 7.,
                                     2.5, 3., 7.,
                                     3.5, 3., 7.,
                                    -2., 4., 8.,
                                    -1., 4., 8.,
                                     0., 4., 8.,
                                     1., 4., 8.,
                                     2., 4., 8.,
                                     3., 4., 8.,
                                     4., 4., 8.,
                                    -2., 3., 5.,
                                    -1., 3., 5.,
                                     1., 3., 5.,
                                     2., 3., 5.]))


        xtarget = torch.cat((xtarget.view(-1, 3), cvt(torch.tensor([0, -0.5, -3])) + xtarget.view(-1, 3)), dim=0).view(-1)

        halfTrain = nTrain // 2

        xInit = cvt(torch.tensor([1,-1,-1])) * xtarget.view(-1,3) + cvt(torch.tensor([0,0,10]))
        xInit = xInit.view(1,-1)

        x0     = xInit  + cvt( var0 * torch.randn(halfTrain, d))
        xmore  = xtarget + cvt(var0 * torch.randn(halfTrain, d))
        x0 = torch.cat((x0, xmore), dim=0)

        # validation samples from rho_0
        x0v     = xInit  + cvt( var0 * torch.randn(halfTrain, d))
        for j in range(nAgents):
            x0[ :,3*j+3:3*(j+1)] = 0. *  x0[ :,3*j+3:3*(j+1)]
            x0v[:,3*j+3:3*(j+1)] = 0. *  x0v[:,3*j+3:3*(j+1)]

        prob = SwarmTraj(xtarget, obstacle='blocks', alph_Q=alph[1],  alph_W=alph[2], r= 0.1)

    elif sData == 'singlequad':

        d = 12
        xtarget = cvt(torch.tensor([2., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))

        init = -1.5
        xInit  = cvt(torch.tensor([init, init, init]))
        x0 = xInit + cvt(var0 * torch.randn(nTrain, 3))
        x0 = pad(x0, [0, d - 3, 0, 0], value=0)
        xInit = pad(xInit.view(1,-1), [0, d - 3, 0, 0], value=0)

        # validation samples from rho_0
        x0v = cvt(torch.tensor([init, init, init]) + var0 * torch.randn(nVal, 3))
        x0v = pad(x0v, [0, d - 3, 0, 0], value=0)

        prob = Quadcopter(xtarget,obstacle=None, alph_Q = 0.0, alph_W = 0.0)

    elif sData=='midcross2':
        nAgents = 2
        d = 2 * nAgents
        xtarget = cvt(torch.tensor([2,2,-2,2]))
        xInit   = cvt(torch.tensor([-2,-2,2,-2])).view(1,-1)
        x0      = cvt(torch.tensor([-2,-2,2,-2]) + var0 * torch.randn(nTrain, d))
        x0v     = cvt(torch.tensor([-2,-2,2,-2]) + var0 * torch.randn(nTrain, d))
        prob    = Cross2D(xtarget, obstacle=None, alph_Q=alph[1], alph_W=alph[2])

    elif sData == 'midcross4':
        nAgents = 4
        d = 2 * nAgents
        xx = torch.linspace(-2, 2, nAgents)
        xtarget = cvt(torch.stack((xx.flip(dims=[0]), 2 * torch.ones(nAgents)), dim=1).reshape(1,-1))
        xInit = cvt(torch.stack((xx, -2 * torch.ones(nAgents)), dim=1).reshape(1,-1))

        x0  = xInit + cvt(var0 * torch.randn(nTrain, d))
        x0v = xInit + cvt(var0 * torch.randn(nVal, d))
        prob = Cross2D(xtarget, obstacle=None, alph_Q=alph[1], alph_W=alph[2], r=0.4)
    elif sData == 'midcross20':
        nAgents = 20
        d = 2 * nAgents
        xx = torch.linspace(-6, 6, nAgents)
        xtarget = cvt(torch.stack((xx.flip(dims=[0]), 6 * torch.ones(nAgents)), dim=1).reshape(1, -1))
        xInit = cvt(torch.stack((xx, -6 * torch.ones(nAgents)), dim=1).reshape(1, -1))

        x0 = xInit + cvt(var0 * torch.randn(nTrain, d))
        x0v = xInit + cvt(var0 * torch.randn(nVal, d))
        prob = Cross2D(xtarget, obstacle=None, alph_Q=alph[1], alph_W=alph[2], r=0.15)
    elif sData == 'midcross30':
        nAgents = 30
        d = 2 * nAgents
        xx = torch.linspace(-6, 6, nAgents) 
        tmp = torch.tensor([6,4,2])
        tmp = tmp.view(-1,1).repeat(nAgents//3,1).view(-1)
        xtarget = cvt(torch.stack((xx.flip(dims=[0]), tmp), dim=1).reshape(1, -1))
        
        tmp = torch.tensor([-6,-4,-2])
        tmp = tmp.view(-1,1).repeat(nAgents//3,1).view(-1)
        xInit = cvt(torch.stack((xx, tmp), dim=1).reshape(1, -1))

        x0 = xInit + cvt(var0 * torch.randn(nTrain, d))
        x0v = xInit + cvt(var0 * torch.randn(nVal, d))
        prob = Cross2D(xtarget, obstacle=None, alph_Q=alph[1], alph_W=alph[2], r=0.2)
    elif sData == 'swap2':
        nAgents = 2
        d = 2 * nAgents
        xtarget = cvt(torch.tensor([10., 0., -10., 0.]))
        xInit = cvt(torch.tensor([-10., 0., 10., 0.])).reshape(1, -1)
        x0  = xInit + cvt(var0 * torch.randn(nTrain, d))
        x0v = xInit + cvt(var0 * torch.randn(nVal, d))
        prob = Cross2D(xtarget, obstacle='hardcorridor', alph_Q=alph[1], alph_W=alph[2], r=1.0)
    elif sData == 'swap12':
        nAgents = 12
        d = 2 * nAgents
        xtarget = cvt(torch.tensor([ 2,2, 0,0,  10,0, -10,0,   5,5, -5,-5,  -4,2, -6,-1,   5,-5, -5,5,   2,-2, -2,-2 ]))
        xInit   = cvt(torch.tensor([0,0, 2,2,  -10,0, 10,0,   -5,-5, 5,5,  -6,-1, -4,2,  -5,5, 5,-5,  -2,-2, 2,-2 ])).reshape(1,-1)
        x0 = xInit + cvt(var0 * torch.randn(nTrain, d))
        x0v = xInit + cvt(var0 * torch.randn(nVal, d))
        prob = Cross2D(xtarget, obstacle=None, alph_Q=alph[1], alph_W=alph[2], r=0.5)
    elif sData == 'swap12_5pair':
        nAgents = 10
        d = 2 * nAgents
        xtarget = cvt(torch.tensor([ 2,2, 0,0,  10,0, -10,0,   5,5, -5,-5,  -4,2, -6,-1,   5,-5, -5,5 ]))
        xInit   = cvt(torch.tensor([0,0, 2,2,  -10,0, 10,0,   -5,-5, 5,5,  -6,-1, -4,2,  -5,5, 5,-5 ])).reshape(1,-1)
        x0 = xInit + cvt(var0 * torch.randn(nTrain, d))
        x0v = xInit + cvt(var0 * torch.randn(nVal, d))
        prob = Cross2D(xtarget, obstacle=None, alph_Q=alph[1], alph_W=alph[2], r=0.5)
    elif sData == 'swap12_4pair':
        nAgents = 8
        d = 2 * nAgents
        xtarget = cvt(torch.tensor([ 2,2, 0,0,  10,0, -10,0,   5,5, -5,-5,  -4,2, -6,-1 ]))
        xInit   = cvt(torch.tensor([0,0, 2,2,  -10,0, 10,0,   -5,-5, 5,5,  -6,-1, -4,2])).reshape(1,-1)
        x0 = xInit + cvt(var0 * torch.randn(nTrain, d))
        x0v = xInit + cvt(var0 * torch.randn(nVal, d))
        prob = Cross2D(xtarget, obstacle=None, alph_Q=alph[1], alph_W=alph[2], r=0.5)
    elif sData == 'swap12_3pair':
        nAgents = 6
        d = 2 * nAgents
        xtarget = cvt(torch.tensor([ 2,2, 0,0,  10,0, -10,0,   5,5, -5,-5 ]))
        xInit   = cvt(torch.tensor([0,0, 2,2,  -10,0, 10,0,   -5,-5, 5,5 ])).reshape(1,-1)
        x0 = xInit + cvt(var0 * torch.randn(nTrain, d))
        x0v = xInit + cvt(var0 * torch.randn(nVal, d))
        prob = Cross2D(xtarget, obstacle=None, alph_Q=alph[1], alph_W=alph[2], r=0.5)
    elif sData == 'swap12_2pair':
        nAgents = 4
        d = 2 * nAgents
        xtarget = cvt(torch.tensor([ 2,2, 0,0,  10,0, -10,0 ]))
        xInit   = cvt(torch.tensor([0,0, 2,2,  -10,0, 10,0 ])).reshape(1,-1)
        x0 = xInit + cvt(var0 * torch.randn(nTrain, d))
        x0v = xInit + cvt(var0 * torch.randn(nVal, d))
        prob = Cross2D(xtarget, obstacle=None, alph_Q=alph[1], alph_W=alph[2], r=0.5)
    elif sData == 'swap12_1pair':
        nAgents = 2
        d = 2 * nAgents
        xtarget = cvt(torch.tensor([ 2,2, 0,0 ]))
        xInit   = cvt(torch.tensor([0,0, 2,2 ])).reshape(1,-1)
        x0 = xInit + cvt(var0 * torch.randn(nTrain, d))
        x0v = xInit + cvt(var0 * torch.randn(nVal, d))
        prob = Cross2D(xtarget, obstacle=None, alph_Q=alph[1], alph_W=alph[2], r=0.5)
    else:
        print("incorrect value passed to --data")
        exit(1)


    return prob, x0, x0v, xInit


def resample(x0, xInit, var0, cvt):
    """
    resample rho_0 for next training batch

    :param x0:    nTrain -by- d tensor, previous training batch
    :param xInit: 1 -by- d tensor, center of rho_0
    :param var0: float, variance of rho_0
    :param cvt:   func, conversion function for typing and device
    :return: nTrain -by- d tensor, training batch
    """
    return xInit + cvt( var0 * torch.randn(*x0.shape) )



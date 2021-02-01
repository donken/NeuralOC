# OCflow.py
import math
import torch
from torch.nn.functional import pad
from src.Phi import *

def OCflow(x, Phi, prob, tspan , nt, stepper="rk4", alph =[1.0,1.0,1.0,1.0,1.0,1.0], intermediates=False, noMean=False ):
    """
        main workhorse of the approach

    :param x:       input data tensor nex-by-d
    :param Phi:     neural network
    :param xtarget: target state for OC problem
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list, the alpha value multipliers
    :param intermediates: if True, return the states and controls instead
    :param noMean: if True, do not compute the mean across samples for Jc and cs
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list of the computed costs
    """
    nex, d = x.shape
    h = (tspan[1]-tspan[0]) / nt

    z = pad(x, [0, 1, 0, 0], value=0)
    P = Phi.getGrad(z)  # initial condition gradPhi = p
    P = P[:, 0:d]

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    # nex - by - (2*d + 4)
    z = torch.cat( (x , torch.zeros(nex,4, dtype=x.dtype,device=x.device)) , 1)

    tk = tspan[0]

    if intermediates: # save the intermediate values as well
        # make tensor of size z.shape[0], z.shape[1], nt
        zFull = torch.zeros( *z.shape , nt+1, device=x.device, dtype=x.dtype)
        zFull[:,:,0] = z
        # hold the controls/thrust and torques on the path
        tmp = prob.calcCtrls(z[:, 0:d], P)
        ctrlFull = torch.zeros(*tmp.shape, nt + 1, dtype=x.dtype,device=x.device)

    for k in range(nt):
        if stepper == 'rk4':
            z = stepRK4(ocOdefun, z, Phi, prob, alph, tk, tk + h)
        elif stepper == 'rk1':
            z = stepRK1(ocOdefun, z, Phi, prob, alph, tk, tk + h)
        tk += h
        if intermediates:
            zFull[:, :, k+1] = z
            tmp = pad(z[:,0:d], [0, 1, 0, 0], value=tk-h)
            p = Phi.getGrad(tmp)[:,0:d]
            ctrlFull[:, :, k + 1] = prob.calcCtrls(z[:, 0:d], p)


    resG = ocG(z[:,0:d], prob.xtarget)
    cG   = 0.5 * torch.sum(resG**2, 1, keepdims=True)

    # compute Phi at final time
    tmp = pad(z[:,0:d], [0, 1, 0, 0], value=tspan[1])
    Phi1 = Phi(tmp)
    gradPhi1 = Phi.getGrad(tmp)[:, 0:d]

    if noMean:
        costL = z[:, -4].view(-1,1)
        costG = cG.view(-1,1)
        costHJt = z[:, -3].view(-1,1)
        costHJf = torch.sum(torch.abs(Phi1 - alph[0] * cG), 1).view(-1,1)
        costHJgrad = torch.sum(torch.abs(gradPhi1 - alph[0] * resG), 1).view(-1,1)
        costQ = z[:, -2].view(-1,1)
        costW = z[:, -1].view(-1,1)
        cs = [costL, costG, costHJt, costHJf, costHJgrad, costQ, costW]
        Jc = costL + alph[0] * costG + alph[3] * costHJt + alph[4] * costHJf + alph[5] * costHJgrad
        return Jc, cs


    # ASSUME all examples are equally weighted
    costL   = torch.mean(z[:,-4])
    costG   = torch.mean(cG)
    costHJt = torch.mean(z[:,-3])
    costHJf = torch.mean(torch.sum(torch.abs(Phi1 - alph[0] * cG), 1))
    costHJgrad = torch.mean(torch.sum(torch.abs(gradPhi1 - alph[0] * resG), 1))
    costQ = torch.mean(z[:, -2])
    costW = torch.mean(z[:, -1])

    cs = [costL, costG, costHJt, costHJf, costHJgrad, costQ, costW]
    # Jc = sum(i[0] * i[1] for i in zip(cs, alph))
    Jc = costL + alph[0]*costG + alph[3]*costHJt + alph[4]*costHJf + alph[5]*costHJgrad

    if intermediates:
        return zFull, ctrlFull
    else:
        return Jc, cs

def ocG(z, xtarget):
    """G for OC problems"""

    d = xtarget.shape[0] # assumes xtarget has only one dimension
    return z[:,0:d] - xtarget


def ocOdefun(x, t, net, prob, alph=None):
    """
    the diffeq function for the 4 ODEs in one

    d_t  [z_x ; L_x ; hjt_x ; dQ_x ; dW_x] = odefun( [z_x ; L_x ; hjt_x ; dQ_x ; dW_x] , t )

    z_x - state
    L_x - accumulated transport costs
    hjt_x - accumulated error between grad_t Phi and H
    dQ_x - accumulated obstacle cost (maintained for printing purposes)
    dW_x - accumulated interaction cost (maintained for printing purposes)

    :param x:    nex -by- d+4 tensor, state of diffeq
    :param t:    float, time
    :param net:  neural network Phi
    :param prob: problem Object
    :param alph: list, the 6 alpha values for the OC problem
    :return:
    """
    nex, d_extra = x.shape
    d = (d_extra - 4)

    z = pad(x[:, :d], (0, 1, 0, 0), value=t)  # concatenate with the time t

    gradPhi = net.getGrad(z)

    L, H, Q, W = prob.calcLHQW(z[:,:d], gradPhi[:,0:d])

    res = torch.zeros(nex,d+4, dtype=x.dtype, device=x.device) # [dx ; dv ; hjt]

    res[:, 0:d]   = - prob.calcGradpH(z[:,:d] , gradPhi[:,0:d]) # dx
    res[:, d]     = L.squeeze()                                 # dv
    res[:, d + 1] = torch.abs( gradPhi[:,-1] - H.squeeze() )    # HJt
    res[:, d + 2] = Q.squeeze()                                 # Q  # included merely for printing
    res[:, d + 3] = W.squeeze()                                 # W  # included merely for printing

    return res   # torch.cat((dx, dv, hjt, f), 1)


def stepRK1(odefun, z, Phi, prob, alph, t0, t1):
    """
        Runge-Kutta 1 / Forward Euler integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 6 alpha values for the OC problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """
    z += (t1 - t0) * odefun(z, t0, Phi, prob, alph=alph)
    return z

def stepRK4(odefun, z, Phi, prob, alph, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 6 alpha values for the OC problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, prob, alph=alph)
    z = z0 + (1.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, prob, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, prob, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + K , t0+h , Phi, prob, alph=alph)
    z += (1.0/6.0) * K

    return z











# compareCorridor.py
# softcorridor plot baseline and NN on same plot and calculate values for table

import matplotlib
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except:
    matplotlib.use('Agg')  # for linux server with no tkinter
    import matplotlib.pyplot as plt

import argparse
import torch
import os
import src.utils as utils
from src.utils import count_parameters
from src.Phi import *
from src.OCflow import OCflow, ocG
from src.initProb import *


parser = argparse.ArgumentParser('Optimal Control')
parser.add_argument("--nt"      , type=int, default=50)
parser.add_argument('--resume'  , type=str, default='experiments/oc/pretrained/softcorridor_nn_checkpt.pth') # NN model
parser.add_argument('--baseline', type=str, default='experiments/oc/pretrained/softcorridor_baseline_checkpt.pth') # baseline model
parser.add_argument('--save'    , type=str, default='experiments/oc/pretrained/eval/')
parser.add_argument('--prec'    , type=str, default='single', choices=['single','double'])
parser.add_argument('--approach', type=str, default='ocflow', choices=['ocflow'])

args = parser.parse_args()

if args.prec =='double':
    argPrec = torch.float64
else:
    argPrec = torch.float32

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device('cpu') # only supports cpu

figNum = 1
lw = 2 # linewidth
strTitle = 'eval_' + os.path.basename(args.resume)[:-12]

fontsize = 18
title_fontsize = 22

if __name__ == '__main__':

    torch.set_default_dtype(argPrec)
    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

    # load model
    logger.info(' ')
    logger.info("loading model: {:}".format(args.resume))
    logger.info(' ')
    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    m    = checkpt['args'].m
    alph = checkpt['args'].alph
    nTh  = checkpt['args'].nTh
    data = checkpt['args'].data

    prob, x0, _, xInit = initProb(data, 10, 11, var0=0.5, alph=alph, cvt=cvt)
    d = x0.size(1)

    net = Phi(nTh=nTh, m=m, d=d, alph=alph)  # the phi aka the value function
    net.load_state_dict(checkpt["state_dict"])
    net = net.to(argPrec).to(device)

    nt = args.nt
    xtarget = prob.xtarget

    strTitle = 'eval_' + os.path.basename(args.resume)[:-12]

    with torch.no_grad():
        net.eval()
        sPath = args.save + '/figs/' + strTitle + '.png'

        # just the xInit point, printed G includes the alpha_0
        prob.eval()
        logger.info('{:8s} {:14s} {:13s} {:13s} {:11s} {:11s} {:11s} {:11s} {:11s} '.format(
            '1 xInit', 'L+G', 'L', 'G', 'HJt', 'HJf', 'HJg', 'Q', 'W'))
        Jc, cs = OCflow(xInit, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph)
        logger.info( 'NN     {:14.6e} {:13.5e} {:13.5e} {:11.3e} {:11.3e} {:11.3e} {:11.3e} {:11.3e}'.format(
             cs[0] + alph[0]*cs[1], cs[0], alph[0]*cs[1], alph[3]*cs[2], alph[4]*cs[3], alph[5]*cs[4], cs[5], cs[6] ))

        x0 = xInit

        nnTraj, nnCtrls = OCflow(x0, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph, intermediates=True)


        # load baseline
        vopt = torch.load(args.baseline, map_location=lambda storage, loc: storage)

        baseTraj = torch.zeros( d, nt + 1)
        h = 1. / nt
        baseTraj[:,0] = x0
        accL = 0 # accumulated along trajectory
        accQ = 0
        accW = 0
        for j in range(nt):
            L, _, Q, W = prob.calcLHQW(baseTraj[:,j].view(1, -1), vopt[j, :].view(1, -1))
            accL = accL + h * L
            accQ = accQ + h * Q
            accW = accW + h * W
            baseTraj[:,j+1] = baseTraj[:,j] + h * vopt[j,:]
        cG = 0.5 * torch.sum(ocG(baseTraj[:,-1].view(1, -1), prob.xtarget) ** 2, 1, keepdims=True)
        G = net.alph[0] * cG
        totLoss = G + accL
        logger.info('{:8s} {:14s} {:13s} {:13s} {:11s} {:11s} '.format('', 'L+G', 'L', 'G', 'Q', 'W'))
        logger.info('base   {:14.6e} {:13.5e} {:13.5e} {:11.3e} {:11.3e}'.format(totLoss.item(), accL.item(), G.item(), accQ.item(), accW.item()))


        #------MAKE PLOT---------------
        nex, d = x0.shape
        msz = None
        LOWX = -3
        HIGHX = 3
        LOWY = -2.5
        HIGHY = 4
        extent = [LOWX, HIGHX, LOWY, HIGHY]
        xtarget = prob.xtarget.view(-1).detach().cpu().numpy()

        # plot the obstacle
        if prob.obstacle is not None:
            if x0.min() < -9:  # the swap2 case
                nk = 500
            else:
                nk = 50
            xx = torch.linspace(LOWX, HIGHX, nk)
            yy = torch.linspace(LOWY, HIGHY, nk)
            grid = torch.stack(torch.meshgrid(xx, yy)).to(x0.dtype).to(x0.device)
            gridShape = grid.shape[1:]
            grid = grid.reshape(2, -1).t()
            Qmap = prob.calcObstacle(grid)
            Qmap = Qmap.reshape(gridShape).t().detach().cpu().numpy()

        nnTraj = nnTraj.detach().cpu().numpy()

        fig = plt.figure(figsize=plt.figaspect(1.0))
        fig.set_size_inches(7, 5)
        ax = plt.axes(xlim=(LOWX, HIGHX), ylim=(LOWY, HIGHY))

        if prob.obstacle is not None:
            ax.imshow(Qmap, cmap='hot', extent=extent, origin='lower')  # add obstacle
        for j in range(prob.nAgents):
            ax.scatter(xtarget[2 * j], xtarget[2 * j + 1], marker='x', color='red', label='target' if j == 0 else "")
            ax.plot(baseTraj[2*j,:], baseTraj[2*j+1,:], 'o-', linewidth=2, markersize=msz, color='gray',
                    label='baseline solution' if j == 0 else "")
            circ = matplotlib.patches.Circle((nnTraj[0, 2 * j, -1], nnTraj[0, 2 * j + 1, -1]),
                                             radius=prob.r, fill=False, color='m',label='space bubble' if j == 0 else "")
            ax.plot(nnTraj[0, 2 * j, :], nnTraj[0, 2 * j + 1, :], 'o-', linewidth=2, markersize=msz, label=' NN agent '+str(j+1))
            ax.add_patch(circ)

        font_sz = 11
        plt.tick_params(labelsize=font_sz, which='both', direction='out')
        plt.legend(loc='upper right', mode='expand', ncol=2, framealpha=1.0, fontsize=font_sz)
        plt.savefig(args.save + 'softcorridor_both.pdf', dpi=300, bbox_inches='tight', pad_inches = 0.0)
        logger.info('image saved to ' + args.save + 'softcorridor_both.pdf' )



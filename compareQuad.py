# compareQuad.py.py
# quadcopter plot baseline and NN for comparison

import matplotlib
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except:
    matplotlib.use('Agg')  # for linux server with no tkinter
    import matplotlib.pyplot as plt

# avoid Type 3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import argparse
import torch
import os

import src.utils as utils
from src.Phi import *
from src.OCflow import OCflow
from src.initProb import *


parser = argparse.ArgumentParser('Optimal Control')
parser.add_argument("--nt"    , type=int, default=50)
parser.add_argument('--resume'  , type=str, default='experiments/oc/pretrained/singlequad_nn_checkpt.pth') # NN model
parser.add_argument('--baseline', type=str, default='experiments/oc/pretrained/singlequad_baseline_checkpt.pth') # baseline model
parser.add_argument('--save'    , type=str, default='experiments/oc/pretrained/eval/')
parser.add_argument('--prec'    , type=str, default='single', choices=['single','double'])
parser.add_argument('--make_vid', default=False, action='store_true', help="including this flag will produce video")

args = parser.parse_args()

if args.prec =='double':
    argPrec = torch.float64
else:
    argPrec = torch.float32

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device('cpu') # only supported on cpu

# definitions
lw  = 2 # linewidth
fontsize = 18
title_fontsize = 22
utils.makedirs(args.save)
strTitle = 'eval_' + os.path.basename(args.resume)[:-12]

figNum = 1
# color options
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def printPlotLarge(fig, sFileName):
    fig.set_size_inches(11, 6.5)
    logger.info('saving as ' + args.save + sFileName)
    fig.savefig(args.save + sFileName, dpi=300 , bbox_inches="tight", pad_inches=0.0)

if __name__ == '__main__':

    torch.set_default_dtype(argPrec)
    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

    # load model
    logger.info(' ')
    logger.info("loading model: {:}".format(args.resume))
    logger.info(' ')
    # reload model
    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    m = checkpt['args'].m
    # alph = args.alph  # if you want to overwrite saved alpha
    alph  = checkpt['args'].alph
    nTh = checkpt['args'].nTh

    data = checkpt['args'].data
    prob, x0, _, xInit = initProb(data, 10, 11, var0=0.5, alph=alph, cvt=cvt)
    d = x0.size(1)

    net = Phi(nTh=nTh, m=m, d=d, alph=alph)  # the phi aka the value function
    net.load_state_dict(checkpt["state_dict"])
    net = net.to(argPrec).to(device)

    nt = args.nt
    xtarget = prob.xtarget

    strTitle = 'eval_' + os.path.basename(args.resume)[:-12]

    bVideo = args.make_vid  # make video
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

        x0 = torch.zeros(10,d)
        x0[0,:] = xInit
        x0[1,0:3] = xInit[0,0:3] + cvt(torch.Tensor([ 0.2, 0.2, 0.1]))
        x0[2,0:3] = xInit[0,0:3] + cvt(torch.Tensor([ 0.2, 0.2,-0.1]))
        x0[3,0:3] = xInit[0,0:3] + cvt(torch.Tensor([ 0.4, 0.0, 0.0]))
        x0[4,0:3] = xInit[0,0:3] + cvt(torch.Tensor([-0.4, 0.0, 0.0]))
        x0[5,0:3] = xInit[0,0:3] + cvt(torch.Tensor([ 0.0, 0.4,-0.3]))
        x0[6,0:3] = xInit[0,0:3] + cvt(torch.Tensor([ 0.0,-0.4,-0.2]))
        x0[7,0:3] = xInit[0,0:3] + cvt(torch.Tensor([ 0.4,-0.4,-0.4]))
        x0[8,0:3] = xInit[0,0:3] + cvt(torch.Tensor([-0.3,-0.5,-0.5]))
        x0[9,0:3] = xInit[0,0:3] + cvt(torch.Tensor([-0.4,-0.8,-0.2]))

        nnTraj, nnCtrls = OCflow(x0, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph, intermediates=True)

        if bVideo:
            sPath = args.save + '/figs/' + strTitle + '.png'
            xviz = torch.zeros(3, d)
            xviz[0, 0:3] = xInit[0, 0:3] + cvt(torch.Tensor([-0.0, 0.0, -0.3]))
            xviz[1, 0:3] = xInit[0, 0:3] + cvt(torch.Tensor([-0.7, 0.7, -0.5]))
            xviz[2, 0:3] = xInit[0, 0:3] + cvt(torch.Tensor([0.8, -0.1, -0.1]))
            videoQuadcopter(xviz, net, prob, nt=nt, sPath=sPath, sTitle="", approach='ocflow')


        # load baseline
        baseline = torch.load(args.baseline, map_location=lambda storage, loc: storage)
        logger.info( 'base   {:14.6e} {:13.5e} {:13.5e} '.format(
            baseline['loss'].item(), baseline['L'].item(), baseline['G'].item() ))

        baseTraj = baseline['traj']
        baseCtrls = baseline['ctrls'].t()


        #------MAKE PLOTS---------------

        # baseline/NN trajectory in 3-D
        # ------------------------------------------------

        # 3-D plot bounds
        xbounds = [-2.7, 2.7]
        ybounds = [-2.7, 2.7]
        zbounds = [-2.7, 2.7]

        for sMethod in ['base','nn']:
            if sMethod=='base':
                traj = baseTraj
            else:
                traj = nnTraj

            fig = plt.figure(figNum)
            ax = plt.axes(projection='3d')

            ax.scatter(xtarget[0].cpu().detach().numpy().item(), xtarget[1].detach().cpu().numpy().item(),
                       xtarget[2].detach().cpu().numpy().item(), s=140, marker='x', c='r',
                       label="target")

            if sMethod == 'base':
                    ax.plot(traj[0, :].view(-1).detach().cpu().numpy(), traj[1, :].view(-1).detach().cpu().numpy(),
                             traj[2, :].view(-1).detach().cpu().numpy(), 'o-')
            else:
                for i in range(traj.shape[0]):
                    ax.plot(traj[i, 0, :].view(-1).detach().cpu().numpy(), traj[i, 1, :].view(-1).detach().cpu().numpy(),
                             traj[i, 2, :].view(-1).detach().cpu().numpy(), 'o-')

            # ax.legend()
            ax.view_init(10, -30)
            ax.set_xlim(*xbounds)
            ax.set_ylim(*ybounds)
            ax.set_zlim(*zbounds)
            ax.tick_params(labelsize=fontsize)
            ax.set_xlabel(r'$x$',fontsize=fontsize, labelpad=25)
            ax.set_ylabel(r'$y$',fontsize=fontsize, labelpad=25)
            ax.set_zlabel(r'$z$',fontsize=fontsize, labelpad=5)

            if sMethod=='base':
                printPlotLarge(fig,  'singlequad_baseline_traj.pdf')
            else:
                printPlotLarge(fig,  'singlequad_nn_traj.pdf')
            figNum += 1



        # plotting path from eagle view
        # ------------------------------------------------
        for sMethod in ['base', 'nn']:
            if sMethod == 'base':
                traj = baseTraj
            else:
                traj = nnTraj

            fig = plt.figure(figNum)
            ax = plt.axes()

            ax.scatter(xtarget[0].cpu().detach().numpy().item(), xtarget[1].detach().cpu().numpy().item(),
                       marker='x', c='r', label="target")

            if sMethod == 'base':
                    ax.plot(traj[0, :].view(-1).detach().cpu().numpy(), traj[1, :].view(-1).detach().cpu().numpy(), 'o-')
            else:
                for i in range(traj.shape[0]):
                    ax.plot(traj[i, 0, :].view(-1).detach().cpu().numpy(), traj[i, 1, :].view(-1).detach().cpu().numpy(), 'o-')

            ax.set_xlim(-2.4, 2.4) # ax.set_xlim(*xbounds)
            ax.set_ylim(-2.4, 2.4) # ax.set_ylim(*ybounds)
            ax.tick_params(labelsize=fontsize)
            ax.set_xlabel(r'$x$',fontsize=fontsize)
            ax.set_ylabel(r'$y$',fontsize=fontsize)
            ax.set_aspect('equal')

            if sMethod=='base':
                printPlotLarge(fig,  'singlequad_baseline_birdview.pdf')
            else:
                printPlotLarge(fig,  'singlequad_nn_birdview.pdf')
            figNum += 1

        # get just the xInit
        nnTraj  = nnTraj[0,:,:]
        nnCtrls = nnCtrls[0,:,:]

        # plotting controls
        # ------------------------------------------------
        fig = plt.figure(figNum)
        ax = plt.axes()

        timet = range(1,nt+1)
        # not using the t=0 values
        ax.plot(timet, nnCtrls[0, 1:].cpu().numpy() , 'o-', label=r'NN $u$',           c=CB_color_cycle[0])
        ax.plot(timet, nnCtrls[1, 1:].cpu().numpy() , 'o-', label=r'NN $\tau_\psi$',   c=CB_color_cycle[5])
        ax.plot(timet, nnCtrls[2, 1:].cpu().numpy() , 'o-', label=r'NN $\tau_\theta$', c=CB_color_cycle[1])
        ax.plot(timet, nnCtrls[3, 1:].cpu().numpy() , 'o-', label=r'NN $\tau_\phi$',   c=CB_color_cycle[7])
        ax.plot(timet, baseCtrls[0, :].cpu().numpy(), 'x-', label=r'Baseline $u$',          c=CB_color_cycle[0])
        ax.plot(timet, baseCtrls[1, :].cpu().numpy(), 'x-', label=r'Baseline $\tau_\psi$',  c=CB_color_cycle[5])
        ax.plot(timet, baseCtrls[2, :].cpu().numpy(), 'x-', label=r'Baseline $\tau_\theta$',c=CB_color_cycle[1])
        ax.plot(timet, baseCtrls[3, :].cpu().numpy(), 'x-', label=r'Baseline $\tau_\phi$',  c=CB_color_cycle[7])
        ax.legend(ncol=2,fontsize=fontsize,loc='upper left', bbox_to_anchor=(0.37, 1.0))
        ax.set_xticks([0, nt / 2, nt])
        ax.set_xticklabels(['0', '0.5', r'$T$=1'])
        ax.set_xlabel(r'Time ($n_t={}$)'.format(nt),fontsize=fontsize)
        ax.set_ylabel('Control',fontsize=fontsize, labelpad=10)
        ax.tick_params(labelsize=fontsize)

        printPlotLarge(fig,  'singlequad_both_ctrls.pdf')
        figNum += 1


        # plotting positional velocity
        #------------------------------------------------
        fig = plt.figure(figNum)
        ax = plt.axes()

        timet = range(nt+1)
        ax.plot(timet, nnTraj[6, :].cpu().numpy(), 'o-', label=r'NN $v_x$', c=CB_color_cycle[0])
        ax.plot(timet, nnTraj[7, :].cpu().numpy(), 'o-', label=r'NN $v_y$', c=CB_color_cycle[5])
        ax.plot(timet, nnTraj[8, :].cpu().numpy(), 'o-', label=r'NN $v_z$', c=CB_color_cycle[7])
        ax.plot(timet, baseTraj[6, :].cpu().numpy(), 'x-', label=r'Baseline $v_x$', c=CB_color_cycle[0])
        ax.plot(timet, baseTraj[7, :].cpu().numpy(), 'x-', label=r'Baseline $v_y$', c=CB_color_cycle[5])
        ax.plot(timet, baseTraj[8, :].cpu().numpy(), 'x-', label=r'Baseline $v_z$', c=CB_color_cycle[7])

        ax.legend(ncol=2,fontsize=fontsize)
        ax.set_xticks([0, nt / 2, nt])
        ax.set_xticklabels(['0', '0.5', r'$T$=1'])
        ax.set_xlabel(r'Time ($n_t={}$)'.format(nt),fontsize=fontsize)
        ax.set_ylabel('Velocity',fontsize=fontsize, labelpad=36) # pad to line up with angles plot
        ax.tick_params(labelsize=fontsize)
        ax.set_xlim(left=0, right=50)

        printPlotLarge(fig,  'singlequad_both_posvel.pdf')
        figNum += 1

        # plotting angular velocity
        # ------------------------------------------------
        fig = plt.figure(figNum)
        ax = plt.axes()

        timet = range(nt+1)
        ax.plot(timet, nnTraj[9,  :].cpu().numpy(), 'o-', label=r'NN $v_\psi$', c=CB_color_cycle[0])
        ax.plot(timet, nnTraj[10, :].cpu().numpy(), 'o-', label=r'NN $v_\theta$', c=CB_color_cycle[5])
        ax.plot(timet, nnTraj[11, :].cpu().numpy(), 'o-', label=r'NN $v_\phi$', c=CB_color_cycle[7])
        ax.plot(timet, baseTraj[9, :].cpu().numpy(), 'x-', label=r'Baseline $v_\psi$', c=CB_color_cycle[0])
        ax.plot(timet, baseTraj[10,:].cpu().numpy(), 'x-', label=r'Baseline $v_\theta$', c=CB_color_cycle[5])
        ax.plot(timet, baseTraj[11,:].cpu().numpy(), 'x-', label=r'Baseline $v_\phi$', c=CB_color_cycle[7])

        ax.legend(ncol=2,fontsize=fontsize)
        ax.set_xticks([0, nt / 2, nt])
        ax.set_xticklabels(['0', '0.5', r'$T$=1'])
        ax.set_xlabel(r'Time ($n_t={}$)'.format(nt),fontsize=fontsize)
        ax.set_ylabel('Velocity',fontsize=fontsize, labelpad=21) # pad to line up with angles plot
        ax.tick_params(labelsize=fontsize)
        ax.set_xlim(left=0, right=50)

        printPlotLarge(fig,  'singlequad_both_angvel.pdf')
        figNum += 1



        # plotting angles
        # ------------------------------------------------
        fig = plt.figure(figNum)
        ax = plt.axes()

        timet = range(nt+1)
        ax.plot(timet, baseTraj[3,:].cpu().numpy(), 'x-', label=r'Baseline $\psi$', c=CB_color_cycle[0])
        ax.plot(timet, baseTraj[4,:].cpu().numpy(), 'x-', label=r'Baseline $\theta$', c=CB_color_cycle[5])
        ax.plot(timet, baseTraj[5,:].cpu().numpy(), 'x-', label=r'Baseline $\phi$', c=CB_color_cycle[7])
        ax.plot(timet, nnTraj[3, :].cpu().numpy() , 'o-', label=r'NN $\psi$', c=CB_color_cycle[0])
        ax.plot(timet, nnTraj[4, :].cpu().numpy() , 'o-', label=r'NN $\theta$', c=CB_color_cycle[5])
        ax.plot(timet, nnTraj[5, :].cpu().numpy() , 'o-', label=r'NN $\phi$', c=CB_color_cycle[7])


        ax.legend(fontsize=fontsize)
        ax.set_xticks([0, nt / 2, nt])
        ax.set_xticklabels(['0', '0.5', r'$T$=1'])
        ax.set_xlabel(r'Time ($n_t={}$)'.format(nt),fontsize=fontsize)
        ax.set_ylabel('Value',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.set_xlim(left=0, right=50)

        printPlotLarge(fig,  'singlequad_both_ang.pdf')
        figNum += 1




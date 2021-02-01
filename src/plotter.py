# plotter.py
# for generating plots

import matplotlib
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except:
    matplotlib.use('Agg')  # for linux server with no tkinter
    import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'inferno'

# avoid Type 3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import os
import torch
from torch.nn.functional import pad
from mpl_toolkits.mplot3d import Axes3D  # for 3D plots
from PIL import Image, ImageDraw # for video

from src.OCflow import OCflow


# the parameters used in this function are shared by most functions in this file
def plotQuadcopter(x, net, prob, nt, sPath, sTitle="", approach='ocflow'):
    """
    plot images of the 12-d quadcopter

    :param x:        tensor, initial spatial point(s) at initial time
    :param net:      Module, the network Phi (or in some cases the baseline)
    :param prob:     problem object, which is needed for targets and obstacles
    :param nt:       int, number of time steps
    :param sPath:    string, path where you want the files saved
    :param sTitle:   string, the title wanted to be applied to the figure
    :param approach: string, used to distinguish how the plot function behaves with inputs 'ocflow' or 'baseline'
    """
    xtarget = prob.xtarget

    if approach == 'ocflow':
        traj, trajCtrl = OCflow(x, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph, intermediates=True)
        trajCtrl = trajCtrl[:,:,1:] # want last dimension to be nt
    elif approach == 'baseline':
        # overload inputs to treat x and net differently for baseline
        traj = x # expects a tensor of size (nex, d, nt+1)
        trajCtrl = net # expects a tensor (nex, a, nt) where a is the dimension of the controls
    else:
        print("approach=" , approach, " is not an acceptable parameter value for plotQuadcopter")

    # 3-d plot bounds
    xbounds = [-3.0, 3.0]
    ybounds = [-3.0, 3.0]
    zbounds = [-3.0, 3.0]

    # make grid of plots
    nCol = 3
    nRow = 2

    fig = plt.figure(figsize=plt.figaspect(1.0))
    fig.set_size_inches(16, 8)
    fig.suptitle(sTitle)

    # positional movement training
    ax = fig.add_subplot(nRow, nCol, 1, projection='3d')
    ax.set_title('Flight Path')

    ax.scatter(xtarget[0].cpu().numpy(), xtarget[1].cpu().numpy(), xtarget[2].cpu().numpy(), s=140, marker='x', c='r',
               label="target")
    for i in range(traj.shape[0]):
        ax.plot(traj[i, 0, :].view(-1).cpu().numpy(), traj[i, 1, :].view(-1).cpu().numpy(),
                traj[i, 2, :].view(-1).cpu().numpy(), 'o-')

    # ax.legend()
    ax.view_init(10, -30)
    ax.set_xlim(*xbounds)
    ax.set_ylim(*ybounds)
    ax.set_zlim(*zbounds)



    # plotting path from eagle view
    ax = fig.add_subplot(nRow, nCol, 2)
    # traj is nex by d+1 by nt+1
    ax.plot(traj[0, 0, :].cpu().numpy(), traj[0, 1, :].cpu().numpy(), 'o-')
    xtarget = xtarget.detach().cpu().numpy()

    ax.scatter(xtarget[0], xtarget[1], marker='x', color='red')
    ax.set_xlim(*xbounds)
    ax.set_ylim(*ybounds)
    ax.set_aspect('equal')
    ax.set_title('Path From Bird View')

    # plot controls
    ax = fig.add_subplot(nRow, nCol, 3)
    timet = range(nt)
    # not using the t=0 values
    ax.plot(timet, trajCtrl[0, 0, :].cpu().numpy(), 'o-', label='u')
    ax.plot(timet, trajCtrl[0, 1, :].cpu().numpy(), 'o-', label=r'$\tau_\psi$')
    ax.plot(timet, trajCtrl[0, 2, :].cpu().numpy(), 'o-', label=r'$\tau_\theta$')
    ax.plot(timet, trajCtrl[0, 3, :].cpu().numpy(), 'o-', label=r'$\tau_\phi$')
    ax.legend()
    ax.set_xticks([0, nt / 2, nt])
    ax.set_xlabel('nt')
    ax.set_ylabel('control')

    # plot L at each time step
    ax = fig.add_subplot(nRow, nCol, 4)
    timet = range(nt)
    # not using the t=0 values
    trajL = torch.sum(trajCtrl[0, :, :] ** 2, dim=0, keepdims=True)
    totL = torch.sum(trajL[0, :]) / nt
    ax.plot(timet, trajL[0, :].cpu().numpy(), 'o-', label='L')
    ax.legend()
    ax.set_xticks([0, nt / 2, nt])
    ax.set_xlabel('nt')
    ax.set_ylabel('L')
    ax.set_title('L(x,T)=' + str(totL.item()))

    # plot velocities
    ax = fig.add_subplot(nRow, nCol, 5)
    timet = range(nt+1)
    ax.plot(timet, traj[0, 6, :].cpu().numpy(), 'o-', label=r'$v_x$')
    ax.plot(timet, traj[0, 7, :].cpu().numpy(), 'o-', label=r'$v_y$')
    ax.plot(timet, traj[0, 8, :].cpu().numpy(), 'o-', label=r'$v_z$')
    ax.plot(timet, traj[0, 9, :].cpu().numpy(), 'o-', label=r'$v_\psi$')
    ax.plot(timet, traj[0, 10, :].cpu().numpy(), 'o-', label=r'$v_\theta$')
    ax.plot(timet, traj[0, 11, :].cpu().numpy(), 'o-', label=r'$v_\phi$')
    ax.legend(ncol=2)
    ax.set_xticks([0, nt / 2, nt])
    ax.set_xlabel('nt')
    ax.set_ylabel('value')

    # plot angles
    ax = fig.add_subplot(nRow, nCol, 6)
    timet = range(nt+1)
    ax.plot(timet, traj[0, 3, :].cpu().numpy(), 'o-', label=r'$\psi$')
    ax.plot(timet, traj[0, 4, :].cpu().numpy(), 'o-', label=r'$\theta$')
    ax.plot(timet, traj[0, 5, :].cpu().numpy(), 'o-', label=r'$\phi$')
    ax.legend()
    ax.set_xticks([0, nt / 2, nt])
    ax.set_xlabel('nt')
    ax.set_ylabel('value')

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()

def videoQuadcopter(x, net, prob, nt, sPath, sTitle="", approach='ocflow'):
    """ make video for the 12-d quadcopter """

    if approach != 'ocflow':
        print('approach ' + approach + ' is not supported')
        return 1

    nex, d = x.shape
    LOWX, HIGHX, LOWY, HIGHY , msz= getMidcrossBounds(x,d)
    extent = [LOWX, HIGHX, LOWY, HIGHY]
    xtarget = prob.xtarget.view(-1).detach().cpu().numpy()

    # 3-d plot bounds
    xbounds = [-3.0, 3.0]
    ybounds = [-3.0, 3.0]
    zbounds = [-3.0, 3.0]

    traj, trajCtrl = OCflow(x, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph, intermediates=True)

    ims = []
    if nex > 1:
        examples = [0,1,2]
    else:
        examples = [0]
    for ex in examples:
        tracePhiFlow = traj[ex, 0:d, :]
        tracePhiFlow = tracePhiFlow.detach().cpu().numpy()
        ctrls = trajCtrl[ex,:,:].detach().cpu().numpy()

        timet = range(1, nt + 1)

        for n in range(1,nt-1):

            # make grid of plots
            nCol = 1
            nRow = 2
            fig = plt.figure(figsize=plt.figaspect(1.0))
            fig.set_size_inches(14, 10)

            # postional movement training
            ax = fig.add_subplot(nRow, nCol, 1, projection='3d')
            ax.set_title('Flight Path')

            for j in range(prob.nAgents):
                for i in range(ex):
                    ax.plot(traj[i, 12*j, :nt-1], traj[i, 12*j+1, :nt-1],
                            traj[i, 12*j+2, :nt-1], '-', linewidth=1, color='gray')
                ax.plot(tracePhiFlow[12 * j, :n], tracePhiFlow[12 * j + 1, :n],
                        tracePhiFlow[12 * j + 2, :n],'o-',  linewidth=2, markersize=msz)
                ax.scatter(xtarget[12 * j], xtarget[12 * j + 1], xtarget[12 * j + 2], s=140, marker='x', color='red')

            ax.view_init(10, -30)
            ax.set_xlim(*xbounds)
            ax.set_ylim(*ybounds)
            ax.set_zlim(*zbounds)

            # plot controls
            ax = fig.add_subplot(nRow, nCol, 2)

            # not using the t=0 values
            ax.plot(timet[:n], trajCtrl[ex, 0, 1:n+1].cpu().numpy(), 'o-', label='u')
            ax.plot(timet[:n], trajCtrl[ex, 1, 1:n+1].cpu().numpy(), 'o-', label=r'$\tau_\psi$')
            ax.plot(timet[:n], trajCtrl[ex, 2, 1:n+1].cpu().numpy(), 'o-', label=r'$\tau_\theta$')
            ax.plot(timet[:n], trajCtrl[ex, 3, 1:n+1].cpu().numpy(), 'o-', label=r'$\tau_\phi$')
            ax.legend(loc='upper center')
            ax.set_xticks([0, nt / 2, nt])
            ax.set_xlabel('nt')
            ax.set_ylabel('control')
            ax.set_ylim(-80,80)
            ax.set_xlim(0,nt)
            ax.set_title('Controls')

            im = fig2img ( fig )
            ims.append(im)
            plt.close(fig)

    sPath = sPath[:-4] + '.gif'
    ims[0].save(sPath, save_all=True, append_images=ims[1:], duration=100, loop=0)
    print('saved video to', sPath)


def videoSwarm(x, net, prob, nt, sPath, sTitle="", approach='ocflow'):
    """ make video for the swarm trajectory planning """
    nex = x.shape[0]
    d = x.shape[1]

    xtarget = prob.xtarget.detach().cpu().numpy()
    msz = 3  # markersize

    if approach == 'ocflow':
        traj, trajCtrl = OCflow(x, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph, intermediates=True)

    # 3-d plot bounds
    xbounds = [   min( x[:, 0::3].min().item() , xtarget[0::3].min().item()) - 1 , max( x[:, 0::3].max().item() , xtarget[0::3].max().item()) + 1 ]
    ybounds = [   min( x[:, 1::3].min().item() , xtarget[1::3].min().item()) - 1 , max( x[:, 1::3].max().item() , xtarget[1::3].max().item()) + 1 ]
    zbounds = [   min( x[:, 2::3].min().item() , xtarget[2::3].min().item()) - 1 , max( x[:, 2::3].max().item() , xtarget[2::3].max().item()) + 1 ]

    xls = torch.linspace(*xbounds, 50)
    yls = torch.linspace(*ybounds, 50)
    gridPts = torch.stack(torch.meshgrid(xls, yls)).to(x.dtype).to(x.dtype).to(x.device)
    gridShape = gridPts.shape[1:]
    gridPts = gridPts.reshape(2, -1).t()

    # setup example initial z
    z0 = pad(gridPts, [0, 1, 0, 0], value=-1.5)
    z0 = pad(z0, [0, 1, 0, 0], value=0.0)

    # make grid of subplots
    nCol = 2
    nRow = 2
    # fig = plt.figure(figsize=plt.figaspect(1.0))
    # fig.set_size_inches(17, 10) # (14,10)
    # fig.suptitle(sTitle)

    ims = []
    for n in range(nt):
        fig = plt.figure()
        fig.set_size_inches(17, 10)

        # positional movement/trajectory
        for place in [1,2,4]: # plot two angles of it
            ax = fig.add_subplot(nRow, nCol, place, projection='3d')
            ax.set_title('Flight Path')

            if prob.obstacle == 'blocks':
                    shade = 0.4
                    # block 1
                    X, Y = np.meshgrid([-2, 2], [-0.5, 0.5])
                    ax.plot_surface(X, Y, 7 * np.ones((2,2)) , alpha=shade, color='gray')
                    ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
                    X, Z = np.meshgrid([-2, 2], [0, 7])
                    ax.plot_surface(X, -0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
                    ax.plot_surface(X,  0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
                    Y, Z = np.meshgrid([-0.5, 0.5], [0, 7])
                    ax.plot_surface(-2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
                    ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
                    # block 2
                    X, Y = np.meshgrid([2, 4], [-1, 1])
                    ax.plot_surface(X, Y, 4 * np.ones((2,2)) , alpha=shade, color='gray')
                    ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
                    X, Z = np.meshgrid([2, 4], [0, 4])
                    ax.plot_surface(X, -1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
                    ax.plot_surface(X,  1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
                    Y, Z = np.meshgrid([-1, 1], [0, 4])
                    ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
                    ax.plot_surface( 4 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')

            for j in range(prob.nAgents):

                ax.plot(traj[0, 3*j  , :n].view(-1).cpu().numpy(),
                        traj[0, 3*j+1, :n].view(-1).cpu().numpy(),
                        traj[0, 3*j+2, :n].view(-1).cpu().numpy(), linewidth=2)
                ax.scatter(xtarget[3*j  ],
                           xtarget[3*j+1],
                           xtarget[3*j+2], s=20, marker='x', c='r', label="target")

            if place == 1:
                ax.view_init(60, -30) # ax.view_init(10, -30)
            elif place==4:
                ax.view_init(20, -5) # ax.view_init(-10, 270)
            else:
                ax.view_init(25, 190)

            # ax.legend()
            ax.set_xlim(*xbounds)
            ax.set_ylim(*ybounds)
            ax.set_zlim(*zbounds)

            # make the panes transparent
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            # make the grid lines transparent
            ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
            ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

        # plotting path from eagle view
        ax = fig.add_subplot(nRow, nCol, 3)
        # traj is nex by d+1 by nt+1
        for j in range(prob.nAgents):
            ax.plot(traj[0, 3*j, :n+1].cpu().numpy(), traj[0, 3*j+2, :n+1].cpu().numpy(), linewidth=2)
            circ = matplotlib.patches.Circle((traj[0, 3*j, n], traj[0, 3*j+2, n]), radius=prob.r, fill=False,color='m')
            ax.add_patch(circ)
            ax.scatter(xtarget[3*j], xtarget[3*j+2], marker='x', color='red')

        if prob.obstacle == 'hardcorridor' or 'blocks' or 'cylinders':
            xls = torch.linspace(*xbounds, 50)
            zls = torch.linspace(*zbounds, 50)
            gridPts = torch.stack(torch.meshgrid(xls, zls)).to(x.dtype).to(x.device)
            gridShape = gridPts.shape[1:]
            gridPts = gridPts.reshape(2, -1).t()

            # setup example initial z
            z0 = pad(gridPts, [0, 1, 0, 0], value=-1.5)
            z0 = pad(z0, [0, 1, 0, 0], value=0.0)

            tmp = torch.zeros(gridPts.shape[0], gridPts.shape[1] + 1)
            tmp[:,0] = gridPts[:,0]
            tmp[:,1] = 0.0 * gridPts[:,0] # fix y=0.0
            tmp[:,2] = gridPts[:, 1]
            Qgrid = prob.calcObstacle(tmp)
            im = ax.imshow(Qgrid.reshape(gridShape).t().detach().cpu().numpy(), extent=[*xbounds, *zbounds], origin='lower')
            # fig.colorbar(im, ax=ax)

        ax.set_xlim(*xbounds)
        ax.set_ylim(*zbounds)
        ax.set_aspect('equal')
        ax.set_title('Path From Side View')

        if not os.path.exists(os.path.dirname(sPath)):
            os.makedirs(os.path.dirname(sPath))
        plt.savefig(sPath, dpi=300)
        plt.close()

        im = fig2img(fig)
        ims.append(im)

    # save video as a gif
    sPath = sPath[:-4] + '.gif'
    ims[0].save(sPath, save_all=True, append_images=ims[1:], duration=50, loop=0)
    print('saved video to', sPath)



def plotSwarm(x, net, prob, nt, sPath, sTitle="", approach='ocflow'):
    """plot images of swarm problem"""

    nex = x.shape[0]
    d = x.shape[1]

    xtarget = prob.xtarget.detach().cpu().numpy()
    msz = 3  # markersize

    if approach == 'ocflow':
        traj, trajCtrl = OCflow(x, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph, intermediates=True)

    # 3-d plot bounds
    xbounds = [   min( x[:, 0::3].min().item() , xtarget[0::3].min().item()) - 1 , max( x[:, 0::3].max().item() , xtarget[0::3].max().item()) + 1 ]
    ybounds = [   min( x[:, 1::3].min().item() , xtarget[1::3].min().item()) - 1 , max( x[:, 1::3].max().item() , xtarget[1::3].max().item()) + 1 ]
    zbounds = [   min( x[:, 2::3].min().item() , xtarget[2::3].min().item()) - 1 , max( x[:, 2::3].max().item() , xtarget[2::3].max().item()) + 1 ]

    xls = torch.linspace(*xbounds, 50)
    yls = torch.linspace(*ybounds, 50)
    gridPts = torch.stack(torch.meshgrid(xls, yls)).to(x.dtype).to(x.device)
    gridShape = gridPts.shape[1:]
    gridPts = gridPts.reshape(2, -1).t()

    # setup example initial z
    z0 = pad(gridPts, [0, 1, 0, 0], value=-1.5)
    z0 = pad(z0, [0, 1, 0, 0], value=0.0)

    # make grid of subplots
    nCol = 3
    nRow = 2
    fig = plt.figure(figsize=plt.figaspect(1.0))
    fig.set_size_inches(17, 10) # (14,10)
    fig.suptitle(sTitle)

    # positional movement/trajectory
    for place in [1,4,5]: # plot multiple angles of it
        ax = fig.add_subplot(nRow, nCol, place, projection='3d')
        ax.set_title('Flight Path')


        if prob.obstacle == 'blocks':
                shade = 0.4
                # block 1
                X, Y = np.meshgrid([-2, 2], [-0.5, 0.5])
                ax.plot_surface(X, Y, 7 * np.ones((2,2)) , alpha=shade, color='gray')
                ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
                X, Z = np.meshgrid([-2, 2], [0, 7])
                ax.plot_surface(X, -0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
                ax.plot_surface(X,  0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
                Y, Z = np.meshgrid([-0.5, 0.5], [0, 7])
                ax.plot_surface(-2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
                ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
                # block 2
                X, Y = np.meshgrid([2, 4], [-1, 1])
                ax.plot_surface(X, Y, 4 * np.ones((2,2)) , alpha=shade, color='gray')
                ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
                X, Z = np.meshgrid([2, 4], [0, 4])
                ax.plot_surface(X, -1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
                ax.plot_surface(X,  1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
                Y, Z = np.meshgrid([-1, 1], [0, 4])
                ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
                ax.plot_surface( 4 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')



        for j in range(prob.nAgents):
            # ax.plot(traj[0, 3*j  , :].view(-1).cpu().numpy(),
            #         traj[0, 3*j+1, :].view(-1).cpu().numpy(),
            #         traj[0, 3*j+2, :].view(-1).cpu().numpy(), 'o-', linewidth=2, markersize=msz)
            # ax.scatter(xtarget[3*j  ],
            #            xtarget[3*j+1],
            #            xtarget[3*j+2], s=140, marker='x', c='r', label="target")
            ax.plot(traj[0, 3*j  , :].view(-1).cpu().numpy(),
                    traj[0, 3*j+1, :].view(-1).cpu().numpy(),
                    traj[0, 3*j+2, :].view(-1).cpu().numpy(), linewidth=2)
            ax.scatter(xtarget[3*j  ],
                       xtarget[3*j+1],
                       xtarget[3*j+2], s=20, marker='x', c='r', label="target")

        if place == 1:
            ax.view_init(60, -30) # ax.view_init(10, -30)
        elif place==4:
            ax.view_init(20, -5) # ax.view_init(-10, 270)
        else:
            ax.view_init(25, 190)

        # ax.legend()
        ax.set_xlim(*xbounds)
        ax.set_ylim(*ybounds)
        ax.set_zlim(*zbounds)

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)


    # plotting path from eagle view
    ax = fig.add_subplot(nRow, nCol, 2)
    # traj is nex by d+1 by nt+1
    for j in range(prob.nAgents):
        ax.plot(traj[0, 3*j, :].cpu().numpy(), traj[0, 3*j+1, :].cpu().numpy(), linewidth=2)
        circ = matplotlib.patches.Circle((traj[0, 3*j, -1], traj[0, 3*j+1, -1]), radius=prob.r, fill=False,color='m')
        ax.add_patch(circ)
        ax.scatter(xtarget[3*j], xtarget[3*j+1], marker='x', color='red')

    if prob.obstacle == 'hardcorridor' or 'blocks' or 'cylinders':
        tmp = z0[:,0:3]
        tmp[:,2] = 1.0 # where altitude z = 1.0
        Qgrid = prob.calcObstacle(tmp)
        im = ax.imshow(Qgrid.reshape(gridShape).t().detach().cpu().numpy(), extent=[*xbounds, *ybounds], origin='lower')
        # fig.colorbar(im, ax=ax)

    ax.set_xlim(*xbounds)
    ax.set_ylim(*ybounds)
    ax.set_aspect('equal')
    ax.set_title('Path From Bird View')

    # plotting path from eagle view
    ax = fig.add_subplot(nRow, nCol, 3)
    # traj is nex by d+1 by nt+1
    for j in range(prob.nAgents):
        ax.plot(traj[0, 3*j, :].cpu().numpy(), traj[0, 3*j+2, :].cpu().numpy(), linewidth=2)
        circ = matplotlib.patches.Circle((traj[0, 3*j, -1], traj[0, 3*j+2, -1]), radius=prob.r, fill=False,color='m')
        ax.add_patch(circ)
        ax.scatter(xtarget[3*j], xtarget[3*j+2], marker='x', color='red')

    if prob.obstacle == 'hardcorridor' or 'blocks' or 'cylinders':
        xls = torch.linspace(*xbounds, 50)
        zls = torch.linspace(*zbounds, 50)
        gridPts = torch.stack(torch.meshgrid(xls, zls)).to(x.dtype).to(x.device)
        gridShape = gridPts.shape[1:]
        gridPts = gridPts.reshape(2, -1).t()

        # setup example initial z
        z0 = pad(gridPts, [0, 1, 0, 0], value=-1.5)
        z0 = pad(z0, [0, 1, 0, 0], value=0.0)

        tmp = torch.zeros(gridPts.shape[0], gridPts.shape[1] + 1, device=x.device, dtype=x.dtype)
        tmp[:,0] = gridPts[:,0]
        tmp[:,1] = 0.0 * gridPts[:,0] # fix y=0.0
        tmp[:,2] = gridPts[:, 1]
        Qgrid = prob.calcObstacle(tmp)
        im = ax.imshow(Qgrid.reshape(gridShape).t().detach().cpu().numpy(), extent=[*xbounds, *zbounds], origin='lower')
        # fig.colorbar(im, ax=ax)

    ax.set_xlim(*xbounds)
    ax.set_ylim(*zbounds)
    ax.set_aspect('equal')
    ax.set_title('Path From Side View')

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


def getMidcrossBounds(x,d):
    LOW = -3
    HIGH = 3

    if d > 20: # for larger number of agents
        LOW  = -6
        HIGH = 6
        msz  = 1 # markersize
    else:
        msz = None

    LOWX = LOW
    LOWY = LOW
    HIGHX = HIGH
    HIGHY = HIGH

    if x.min() < -9: # the swap2 case
        LOWX  = -12
        HIGHX = 12
        LOWY  = -5
        HIGHY = 5
        if d > 20:  # swap12 case
            LOWY = -6
            HIGHY = 6
            msz = 0  # markersize

    return LOWX, HIGHX, LOWY, HIGHY, msz


def plotMidCross(x, net, prob, nt, sPath, sTitle="", approach='ocflow'):
    """ plot images for the midcross and swap problem objects"""
    fontsize = 20

    nex, d = x.shape
    msz = None

    LOWX, HIGHX, LOWY, HIGHY, msz = getMidcrossBounds(x,d)

    # #################### TEMPORARY, DELETE LATER
    # LOWX = LOWX - 2
    # HIGHX = HIGHX + 2
    # LOWY = LOWY - 2
    # HIGHY = 3

    extent = [LOWX, HIGHX, LOWY, HIGHY]

    if approach == 'ocflow':
        traj, trajCtrl = OCflow(x, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph, intermediates=True)
    elif approach == 'baseline':
        # treat ``net'' differently
        traj = net


    # ------------------------------------------------------------------
    # make plots

    fig, allAx = plt.subplots(1, 7) # make one axis
    fig.set_size_inches(24, 3) # (24,4)
    # fig.suptitle(sTitle)
    plt.subplots_adjust(wspace=0.1)

    # plot the obstacle
    if prob.obstacle is not None:
        if x.min() < -9:  # the swap2 case
            nk = 500
        else:
            nk = 50
        xx = torch.linspace(LOWX, HIGHX, nk)
        yy = torch.linspace(LOWY, HIGHY, nk)
        grid = torch.stack(torch.meshgrid(xx, yy)).to(x.dtype).to(x.device)
        gridShape = grid.shape[1:]
        grid = grid.reshape(2, -1).t()
        Qmap = prob.calcObstacle(grid)
        Qmap = Qmap.reshape(gridShape).t().detach().cpu().numpy()

    # # obstacle (for phi flow)
    # ax = allAx[0, 0]
    # ax.set_title('Phi Flow')

    # plot paths of a few points
    # for ex in range(nex):  # for each time point
    ex = 0
    tracePhiFlow = traj[ex, 0:d, :]
    tracePhiFlow = tracePhiFlow.detach().cpu().numpy()

    t = nt // 6 # nt // 8
    mid = nt // 2
    tsteps = [t , 2*t, 3 * t, mid, 4* t, 5 * t, nt]
    for i, n in enumerate(tsteps):
        ax = allAx[i]
        ax.set_title(r'$n_t$=' + str(n), fontsize=fontsize)

        for j in range(prob.nAgents):
            ax.plot(tracePhiFlow[2 * j, :n], tracePhiFlow[2 * j + 1, :n], 'o-', linewidth=2, markersize=msz)
            circ = matplotlib.patches.Circle((tracePhiFlow[2 * j, n - 1], tracePhiFlow[2 * j + 1, n - 1]),
                                             radius=prob.r, fill=False, color='m')
            ax.add_patch(circ)

    xtarget = prob.xtarget.view(-1).detach().cpu().numpy()

    for i in range(allAx.shape[0]):
        for na in range(prob.nAgents):
            allAx[i].scatter(xtarget[2 * na], xtarget[2 * na + 1], marker='x', color='red')
        if prob.obstacle is not None:
            allAx[i].imshow(Qmap, cmap='hot', extent=extent, origin='lower')  # add a obstacle

    if x.min() < -9:  # the swap case
        for i in range(allAx.shape[0]):
            allAx[i].tick_params(labelsize=fontsize - 2, which='both', direction='out')
            if i>0:
                allAx[i].get_yaxis().set_visible(False)
            allAx[i].set_aspect('auto')
    else:
        for i in range(allAx.shape[0]):
            allAx[i].tick_params(labelsize=fontsize-2, which='both', direction='out')
            if i>0:
                allAx[i].get_yaxis().set_visible(False)
            # allAx[i].get_xaxis().set_visible(False)

            allAx[i].set_aspect('equal')

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300)
    plt.close()


def plotMidCrossJustFinal(x, net, prob, nt, sPath, sTitle="", approach='ocflow'):
    """ for when we just want the final time image/full path of the solutions """
    fontsize = 20

    nex, d = x.shape
    msz = None

    leg_font_sz = 11
    LOWX, HIGHX, LOWY, HIGHY, msz = getMidcrossBounds(x,d)

    # #################### TEMPORARY, for post majorshock
    # LOWX = LOWX - 2
    # HIGHX = HIGHX + 2
    # LOWY = LOWY - 2
    # HIGHY = 3

    extent = [LOWX, HIGHX, LOWY, HIGHY]

    if approach == 'ocflow':
        traj, trajCtrl = OCflow(x, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph,
                                       intermediates=True)
        # traj has shape     ( nex , d+1, nt+1)
        # gradTraj has shape ( nex , d+1, nt+1)
    elif approach == 'baseline':
        # treat ``net'' differently
        traj = net


    # ------------------------------------------------------------------
    # make plots

    # fig, ax = plt.subplots(1, 1) # make one axis
    fig = plt.figure()
    ax = plt.axes(xlim=(LOWX, HIGHX), ylim=(LOWY, HIGHY))
    # fig.set_size_inches(24, 3) # (24,4)
    # fig.suptitle(sTitle)
    plt.subplots_adjust(wspace=0.1)

    # plot the obstacle
    if prob.obstacle is not None:
        if x.min() < -9:  # the swap2 case
            nk = 500
        else:
            nk = 50
        xx = torch.linspace(LOWX, HIGHX, nk)
        yy = torch.linspace(LOWY, HIGHY, nk)
        grid = torch.stack(torch.meshgrid(xx, yy)).to(x.dtype).to(x.device)
        gridShape = grid.shape[1:]
        grid = grid.reshape(2, -1).t()
        Qmap = prob.calcObstacle(grid)
        Qmap = Qmap.reshape(gridShape).t().detach().cpu().numpy()

    # plot paths of a few points
    # for ex in range(nex):  # for each time point
    ex = 0
    tracePhiFlow = traj[ex, 0:d, :]
    tracePhiFlow = tracePhiFlow.detach().cpu().numpy()

    n=nt

    for j in range(prob.nAgents):
        ax.plot(tracePhiFlow[2 * j, :n], tracePhiFlow[2 * j + 1, :n], 'o-', linewidth=2, markersize=msz)
        circ = matplotlib.patches.Circle((tracePhiFlow[2 * j, n - 1], tracePhiFlow[2 * j + 1, n - 1]),
                                         radius=prob.r, fill=False, color='m')
        ax.add_patch(circ)

    xtarget = prob.xtarget.view(-1).detach().cpu().numpy()

    for na in range(prob.nAgents):
        ax.scatter(xtarget[2 * na], xtarget[2 * na + 1], marker='x', color='red')
    if prob.obstacle is not None:
        ax.imshow(Qmap, cmap='hot', extent=extent, origin='lower')  # add a obstacle

    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    plt.savefig(sPath, dpi=300, bbox_inches='tight', pad_inches = 0.0)
    plt.close()



def videoMidCross(x, net, prob, nt, sPath, sTitle="", approach='ocflow', shockspec=None):
    """
    make video for midcross and swap problem classes

    :param shockspec: list of length 2, shockspec[0] is the time t when the shock occurs
                                        shockspec[1] is the d-dim tensor shift
    """

    if approach != 'ocflow':
        print('approach ' + approach + ' is not supported')
        return 1

    leg_font_sz = 11

    nex, d = x.shape
    LOWX, HIGHX, LOWY, HIGHY , msz= getMidcrossBounds(x,d)
    extent = [LOWX, HIGHX, LOWY, HIGHY]
    xtarget = prob.xtarget.view(-1).detach().cpu().numpy()

    # plot the obstacle
    if prob.obstacle is not None:
        if x.min() < -9:  # the swap2 case, we want a finer grid for plotting the obstacle
            nk = 500
        else:
            nk = 50
        xx = torch.linspace(LOWX, HIGHX, nk)
        yy = torch.linspace(LOWY, HIGHY, nk)
        grid = torch.stack(torch.meshgrid(xx, yy)).to(x.dtype).to(x.device)
        gridShape = grid.shape[1:]
        grid = grid.reshape(2, -1).t()
        Qmap = prob.calcObstacle(grid)
        Qmap = Qmap.reshape(gridShape).t().detach().cpu().numpy()

    if shockspec is None:
        traj, trajCtrl = OCflow(x, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph, intermediates=True)
        ims = []

        # choose how many examples to plot
        if nex > 1:
            examples = [0,1,2]
        else:
            examples = [0]
        for ex in examples:
            tracePhiFlow = traj[ex, 0:d, :]
            tracePhiFlow = tracePhiFlow.detach().cpu().numpy()
            for n in range(1,nt):
                fig = plt.figure()
                ax = plt.axes(xlim=(LOWX, HIGHX), ylim=(LOWY, HIGHY))

                if prob.obstacle is not None:
                    ax.imshow(Qmap, cmap='hot', extent=extent, origin='lower')  # add obstacle
                for j in range(prob.nAgents):
                    for i in range(ex):
                        ax.plot(traj[i, 2*j, :], traj[i, 2*j+1, :], '-', linewidth=1, color='gray')
                    ax.plot(tracePhiFlow[2 * j, :n], tracePhiFlow[2 * j + 1, :n], 'o-', linewidth=2, markersize=msz)
                    circ = matplotlib.patches.Circle((tracePhiFlow[2 * j, n - 1], tracePhiFlow[2 * j + 1, n - 1]),
                                                     radius=prob.r, fill=False, color='m')
                    ax.add_patch(circ)
                    ax.scatter(xtarget[2 * j], xtarget[2 * j + 1], marker='x', color='red')


                im = fig2img ( fig )
                ims.append(im)
                plt.close(fig)


        sPath = sPath + '.gif'
        ims[0].save(sPath, save_all=True, append_images=ims[1:], duration=100, loop=0)
        print('saved video to', sPath)

    # make a video with a shock in it
    else:
        precShock = shockspec[0]
        nShock = int(precShock * nt)
        shock = shockspec[1]

        traj1, _ = OCflow(x, net, prob, tspan=[0.0, precShock], nt=nShock, stepper="rk4", alph=net.alph, intermediates=True)
        xshocked = traj1[:, :d, -1] + shock
        traj2, _ = OCflow(xshocked, net, prob, tspan=[precShock, 1.0], nt=1+nt-nShock, stepper="rk4", alph=net.alph, intermediates=True)
        trace = torch.cat( (traj1[:,0:d, :], traj2[:,:d, :]),dim=2)
        trace = trace.detach().cpu().numpy()

        LOWX  = LOWX - 2
        HIGHX = HIGHX + 2
        LOWY  = LOWY-2
        HIGHY = HIGHY+3

        xx = torch.linspace(LOWX, HIGHX, 2*nk)
        yy = torch.linspace(LOWY, HIGHY, 2*nk)
        grid = torch.stack(torch.meshgrid(xx, yy)).to(x.dtype).to(x.device)
        gridShape = grid.shape[1:]
        grid = grid.reshape(2, -1).t()
        Qmap = prob.calcObstacle(grid)
        Qmap = Qmap.reshape(gridShape).t().detach().cpu().numpy()

        ims = []
        ex = 0

        # hardcoded variance of training space
        trainGauss = torch.Tensor([[-2, -2, 2, -2]]) + 1.0 * torch.randn(10000, 4)
        print('assuming variance = 1.0')

        # to make vector graphics figure with reduced size and efficient loading,
        # save the cloud of initial points as a background png, and create vector graphic on top of it

        # save cloud
        fig = plt.figure()
        ax = plt.axes(xlim=(LOWX, HIGHX), ylim=(LOWY, HIGHY))
        if prob.obstacle is not None:
            ax.imshow(Qmap, cmap='hot', extent=[LOWX, HIGHX, LOWY, HIGHY], origin='lower')  # add an obstacle
        for j in range(prob.nAgents):
            p = ax.plot(trace[ex, 2 * j, 0], trace[ex, 2 * j + 1, 0], 'o', linewidth=3, markersize=msz, alpha=0.4)
            ax.scatter(trainGauss[:, 2 * j], trainGauss[:, 2 * j + 1], marker='o', alpha=0.03, color=p[0].get_color())
        # print cloud to a png
        plt.axis('off')
        plt.savefig(os.path.dirname(sPath) + '/cloud.png', dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

        # load cloud png as the background
        img = plt.imread(os.path.dirname(sPath) + '/cloud.png')

        for n in range(1, trace.shape[2]):
            fig, ax = plt.subplots()
            ax.imshow(img, extent=[LOWX,HIGHX,LOWY,HIGHY])
            for j in range(prob.nAgents):
                # need to grab the color and need a stronger alpha for the legend
                p = ax.plot(trace[ex, 2 * j, 0], trace[ex, 2 * j + 1, 0], 'o', linewidth=3, markersize=msz, alpha=0.4, label='train agent '+str(j+1))

                ax.scatter(xtarget[2*j], xtarget[2*j+1], marker='x', color='red', label='target' if j == 0 else "")

                if n <= nShock:
                    ax.plot(trace[ex, 2 * j, :n], trace[ex, 2 * j + 1, :n], 'o-', color=p[0].get_color(),
                            linewidth=2, markersize=msz, label='agent '+str(j+1))
                    circ = matplotlib.patches.Circle((trace[ex, 2 * j, n - 1], trace[ex, 2 * j + 1, n - 1]),
                                    radius=prob.r, fill=False, color='m',label='space bubble' if j == 0 else "")
                    ax.add_patch(circ)
                    plt.legend(loc='upper right', mode='expand', ncol=2, framealpha=1.0, fontsize=leg_font_sz)
                    plt.tick_params(labelsize=leg_font_sz, which='both', direction='out')
                else:
                    # show interference/shock to system
                    x1 = trace[ex, 2 * j, nShock]
                    y1 = trace[ex, 2 * j + 1, nShock]
                    x2 = trace[ex, 2 * j, nShock] + shock[0, 2 * j]
                    y2 = trace[ex, 2 * j + 1, nShock] + shock[0, 2 * j + 1]
                    annotate = ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(facecolor='red', shrink=0.05),
                                label='shock' if j == prob.nAgents-1 else "")
                    ax.plot(trace[ex, 2 * j, :nShock+1], trace[ex, 2 * j + 1, :nShock+1], 'o-', linewidth=2, markersize=msz,
                            color=p[0].get_color(),label='agent '+str(j+1))
                    ax.plot(trace[ex, 2 * j, nShock+1:n], trace[ex, 2 * j + 1, nShock+1:n], 'o-', linewidth=2, markersize=msz, color=p[0].get_color())

                    # space bubble around agent
                    circ = matplotlib.patches.Circle((trace[ex, 2 * j, n - 1], trace[ex, 2 * j + 1, n - 1]),
                                    radius=prob.r, fill=False, color='m',label='space bubble' if j == 0 else "")
                    ax.add_patch(circ)

                    handles,labels = ax.get_legend_handles_labels()
                    plt.legend(handles = handles + [annotate.arrow_patch], labels = labels+["shock"] ,
                       loc='upper right', mode='expand', ncol=2, framealpha=1.0, fontsize=leg_font_sz)

                    plt.tick_params(labelsize=leg_font_sz, which='both', direction='out')

            im = fig2img ( fig )
            ims.append(im)

            # print final image
            if n==trace.shape[2]-1:
                plt.savefig(sPath + '.pdf', dpi=300,bbox_inches = 'tight', pad_inches = 0.0)
                print('saved final plot to ' + sPath + '.pdf')
            plt.close(fig)

        # save video as a gif
        sPath =  sPath + '.gif'
        ims[0].save(sPath, save_all=True, append_images=ims[1:], duration=100, loop=0)
        print('saved video to', sPath)


# for making a video
# from https://web-backend.icare.univ-lille.fr/tutorials/convert_a_matplotlib_figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


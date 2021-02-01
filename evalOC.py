# evalOC.py
# run a trained model

import argparse
import torch
import os

import src.utils as utils
from src.Phi import *
from src.OCflow import OCflow
from src.plotter import *
from src.initProb import *

parser = argparse.ArgumentParser('Optimal Control')
parser.add_argument("--nt"    , type=int, default=50, help="number of time steps")
parser.add_argument('--alph'  , type=str, default='1.0, 1.0, 1.0, 1.0, 1.0, 1.0')
parser.add_argument('--resume'  , type=str, default='experiments/oc/pretrained/softcorridor_nn_checkpt.pth')
parser.add_argument('--save'    , type=str, default='experiments/oc/eval')
parser.add_argument('--prec'    , type=str, default='single', choices=['single','double'], help="single or double precision")
parser.add_argument('--approach', type=str  , default='ocflow', choices=['ocflow'])
parser.add_argument('--make_vid', default=False, action='store_true', help="including this flag will produce video")
parser.add_argument('--do_shock', default=False, action='store_true', help="including this flag will incorporate shocks")

args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

if args.prec =='double':
    argPrec = torch.float64
else:
    argPrec = torch.float32

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device('cpu') # only support cpu


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
    # alph = args.alph  # overwrite saved alpha
    alph  = checkpt['args'].alph
    nTh = checkpt['args'].nTh

    data = checkpt['args'].data
    prob, x0, _, xInit = initProb(data, 10, 11, var0=1.0, alph=alph, cvt=cvt)
    prob.eval()
    d = x0.size(1)

    net = Phi(nTh=nTh, m=m, d=d, alph=alph)  # the phi aka the value function
    net.load_state_dict(checkpt["state_dict"])
    net = net.to(argPrec).to(device)

    nt = args.nt

    strTitle = 'eval_' + os.path.basename(args.resume)[:-12]

    bJustXinit = True   # eval model just on xInit
    bVideo     = args.make_vid  # make video
    bShock     = args.do_shock  # make video and picture of the shocked system
    with torch.no_grad():
        net.eval()
        sPath = args.save + '/figs/' + strTitle + '.pdf'

        if bJustXinit:
            # just the xInit point, printed G includes the alpha_0
            logger.info('{:8s} {:12s} {:11s} {:11s} {:11s} {:11s} {:11s} {:11s} {:11s} '.format(
                'just xInit', 'L+G', 'L', 'G w/ a0', 'HJt', 'HJfin', 'HJgrad', 'Q', 'W'))
            Jc, cs = OCflow(xInit, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph)

            zFull, ctrlFull = OCflow(xInit, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph, intermediates=True)
            logger.info( '         {:12.4e} {:11.3e} {:11.3e} {:11.3e} {:11.3e} {:11.3e} {:11.3e} {:11.3e}'.format(
                 cs[0] + alph[0]*cs[1], cs[0], alph[0]*cs[1], alph[3]*cs[2], alph[4]*cs[3], alph[5]*cs[4], cs[5], cs[6] ))

            if data == 'softcorridor' or data[0:8] == 'midcross' or data[0:4] == 'swap':
                plotMidCrossJustFinal(xInit, net, prob, nt, sPath, sTitle=strTitle, approach=args.approach)
            elif data == 'swarm50':
                plotSwarm(xInit, net, prob, nt, sPath, sTitle=strTitle, approach='ocflow')
            elif data == 'singlequad':
                plotQuadcopter(xInit, net, prob, nt, sPath, sTitle=strTitle, approach='ocflow')
            else:
                logger.info("plotting not implemented for the provided data")
            logger.info('plot saved to ' + sPath)

        if bVideo:

            if data == 'swarm50':
                videoSwarm(xInit, net, prob, nt, sPath, sTitle=strTitle, approach='ocflow')
            elif data == 'singlequad':
                videoQuadcopter(xInit, net, prob, nt, sPath, sTitle=strTitle, approach='ocflow')
            elif data[0:8] == 'midcross' or data[0:4] == 'swap':
                videoMidCross(xInit, net, prob, nt, sPath[:-4], sTitle=strTitle, approach='ocflow', shockspec=None)
            elif data == 'softcorridor':
                offset = torch.Tensor([[0.4, -0.4, 0.4, 0.4]])
                x0 = torch.cat((xInit - offset, xInit + offset, xInit), dim=0)
                videoMidCross(x0, net, prob, nt, sPath[:-4], sTitle=strTitle, approach='ocflow', shockspec=None)
            else:
                logger.info("video not implemented for the provided data")


        if bShock:

            if data == 'softcorridor':
                # time of shock, then 4-d spatial shock
                shock = [0.1   , torch.Tensor([-0.2,-0.7,-0.0,-0.6]).view(1,-1) ]  # minor shock
                videoMidCross(xInit, net, prob, nt, sPath[:-4]+'_shock', sTitle=strTitle, approach='ocflow', shockspec=shock)
                shock = [0.1  ,  torch.Tensor([ -1.4, -1.0, -5.2, -2.8]).view(1, -1)] # major shock
                videoMidCross(xInit, net, prob, nt, sPath[:-4]+'_majorshock', sTitle=strTitle, approach='ocflow', shockspec=shock)
            else:
                logger.info("shock videos only implemented for softcorridor")



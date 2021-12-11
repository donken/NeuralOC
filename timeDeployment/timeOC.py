# timeOC.py
# time a trained model

import argparse
import torch
import os
import time
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
parser.add_argument('--time_log', type=str, default='time_log', help='file to print timing the deployment/online mode')

args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

if args.prec =='double':
    argPrec = torch.float64
else:
    argPrec = torch.float32


device = torch.device('cpu') # only support cpu





if __name__ == '__main__':

    torch.set_default_dtype(argPrec)
    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

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

    # open time_log to print timing results to
    timeFile = open(args.time_log, 'a')
    print('\n----------\n', file=timeFile)
    print('NN', file=timeFile)
    print('problem: ', data, file=timeFile)
    print('args: ', checkpt['args'], file=timeFile)
    print('device: ', device, file=timeFile)


    net = Phi(nTh=nTh, m=m, d=d, alph=alph)  # the phi aka the value function
    net.load_state_dict(checkpt["state_dict"])
    net = net.to(argPrec).to(device)

    nt = args.nt

    with torch.no_grad():
        net.eval()


        # -------TIME THE DEPLOYED MODEL
        timeTot = 0.0
        start = time.time()
        Jc, cs = OCflow(xInit, net, prob, tspan=[0.0, 1.0], nt=nt, stepper="rk4", alph=net.alph)
        end = time.time()
        timeTot = timeTot + (end-start)
        print('time: %5f   avg time / RK4 timestep: %5f' % (end - start, timeTot / nt), file=timeFile)

        timeFile.close() # close timing file



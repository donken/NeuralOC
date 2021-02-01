# baselineQuad.py
# baseline approach for Quadcopter problem


import matplotlib
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except:
    matplotlib.use('Agg')  # for linux server with no tkinter
    import matplotlib.pyplot as plt
import os
import torch
import argparse

from src.problem.Quadcopter import *
from src.initProb import *
from src.plotter import plotQuadcopter



parser = argparse.ArgumentParser('Baseline')
parser.add_argument('--data', choices=['singlequad'],type=str, default='singlequad')
parser.add_argument("--nt"    , type=int, default=50, help="number of time steps")
parser.add_argument('--alph'  , type=str, default='5000.0, 0.0, 0.0')
# alphas: G, Q (obstacle), W (interaction)
parser.add_argument('--niters', type=int, default=600)
parser.add_argument('--gpu'   , type=int, default=0, help="send to specific gpu")
parser.add_argument('--prec'  , type=str, default='single', choices=['single','double'], help="single or double precision")
parser.add_argument('--save'  , type=str, default='experiments/oc/baseline', help="define the save directory")

args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

if args.prec =='double':
    argPrec = torch.float64
else:
    argPrec = torch.float32

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

nt = args.nt

# quadcopter dynamics
def dyn(ctrls, x, prob):

  f7,f8,f9 = prob.f(x[3:6].unsqueeze(0))
  tmp = (ctrls[0]/prob.mass)

  return torch.cat([x[6:], tmp*f7, tmp*f8, tmp*f9-prob.grav, ctrls[1:4]])


def compute_loss(ctrls, x0, prob, alphG=5000):

  # int_0^T L dt + G
  nt = ctrls.shape[0]
  h = 1.0 / nt

  # compute L(x,T) and propagate x at same time
  J = 0.0
  x = x0
  for i in range(nt):
    dx = dyn(ctrls[i, :], x, prob)
    x = x + h * dx
    J = J + h * ( 2 + torch.norm(ctrls[i,:],p=2)**2  )

  # add G( z(T) )
  J += alphG * 0.5 * torch.norm( x - prob.xtarget , p=2)**2

  return J

def trainBaseline(z0, prob, alphG=5000, nt=50, nIters=10000):
  ctrls = cvt(1.e-2 * torch.randn(nt, 4))
  ctrls = torch.nn.Parameter(ctrls)

  optim = torch.optim.LBFGS([{'params': ctrls}], max_iter=nIters, max_eval=10000, line_search_fn="strong_wolfe", tolerance_grad=1e-05, tolerance_change=1e-06)

  def closure():
      optim.zero_grad()
      err = compute_loss(ctrls, z0, prob, alphG)  # calc loss
      print('loss:', err.item())
      err.backward()
      return err

  optim.step(closure)
  return ctrls


if __name__ == '__main__':

  alphG = args.alph[0]
  prob, _, _, xInit = initProb(args.data, 10, 10, var0=1.0, cvt=cvt,
                               alph=[alphG, args.alph[1], args.alph[2], 0.0, 0.0, 0.0])
  x0 = xInit.squeeze()

  ctrls = trainBaseline(x0, prob, alphG=alphG, nt=50, nIters=16000)
  strTitle = 'baseline_quadcopter_alph{:}_{:}_{:}'.format(int(alphG), int(args.alph[1]), int(args.alph[2]))

  # training complete

  # int_0^T L dt + G
  nt = ctrls.shape[0]
  h = 1.0 / nt

  # compute L(x,T) and propagate x at same time
  J = 0.0
  traj  = cvt(torch.zeros(x0.shape[0],nt+1))
  trajL = cvt(torch.zeros(1,nt+1))
  x = x0
  traj[:,0] = x0
  for i in range(nt):
    dx = dyn(ctrls[i, :], x, prob)
    x = x + h * dx
    traj[:,i+1] = x
    trajL[:,i+1] =  2 + torch.norm(ctrls[i,:],p=2)**2   # off by one error????
    J = J + h * trajL[:,i+1]

  totL = J
  # add G
  G = alphG * 0.5 * torch.norm( x - prob.xtarget , p=2)**2
  loss = totL+G

  print("loss: ", (loss).item(), " L(x,T): " , totL.item() , "  G: ", G.item())

  traj  = traj.detach()
  ctrls = ctrls.detach()

  # save weights
  torch.save({
    'ctrls': ctrls,
    'traj': traj,
    'loss': loss,
    'L':   totL,
    'G':    G
  }, 'experiments/oc/baseline/' + strTitle + '.pth')

  sPath = 'experiments/oc/baseline/figs/' + strTitle +'.png'
  plotQuadcopter(traj.unsqueeze(0), ctrls.t().unsqueeze(0), prob, nt, sPath, sTitle="", approach='baseline')
  print('figure saved to ', sPath)





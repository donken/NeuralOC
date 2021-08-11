# NeuralOC
Pytorch implementation of our neural network approach for solving optimal control problems

![](https://imgur.com/I1HaXmz.gif)

Additional videos at https://imgur.com/a/eWr6sUb
## Associated Publications


#### A Neural Network Approach for High-Dimensional Optimal Control
Full-Paper: https://arxiv.org/abs/2104.03270 

Please cite as

    @article{onken2020neuraloc,
        title   = {A Neural Network Approach for High-Dimensional Optimal Control}, 
        author  = {Derek Onken and Levon Nurbekyan and Xingjian Li and Samy Wu Fung and Stanley Osher and Lars Ruthotto},
        year    = {2021},
        journal = {arXiv:2104.03270},
    }



#### A Neural Network Approach Applied to Multi-Agent Optimal Control
Conference Paper: https://arxiv.org/abs/2011.04757

(Forthcoming at the 2021 European Control Conference)

Please cite as
    
    @article{onken2020neuralmulti,
        title   = {A Neural Network Approach Applied to Multi-Agent Optimal Control}, 
        author  = {Derek Onken and Levon Nurbekyan and Xingjian Li and Samy Wu Fung and Stanley Osher and Lars Ruthotto},
        year    = {2020},
        journal = {arXiv:2011.04757},
    }




## Set-up

Install all the requirements (designed for python 3.6.9):
```
pip install -r requirements.txt 
```

Try a simple command
```
python trainOC.py
```

More elaborate instructions available at [detailedSetup.md](detailedSetup.md).

## Experiments

### Neural Network Examples

#### Pretrained neural network examples

Corridor Experiment
```
python evalOC.py --nt 50 --resume experiments/oc/pretrained/softcorridor_nn_checkpt.pth
  
  # additional flag to produce the shocks for the corridor problem
python evalOC.py --nt 50 --resume experiments/oc/pretrained/softcorridor_nn_checkpt.pth --do_shock
  
  # additional flag to produce video
python evalOC.py --nt 50 --resume experiments/oc/pretrained/softcorridor_nn_checkpt.pth --make_vid
```

2-agent Swap Experiment
```
python evalOC.py --nt 50 --resume experiments/oc/pretrained/swap2_nn_checkpt.pth --make_vid
```

12-agent Swap Experiment
```
python evalOC.py --nt 50 --resume experiments/oc/pretrained/swap12_nn_checkpt.pth --make_vid
```

Swarm Experiment
```
python evalOC.py --nt 80 --resume experiments/oc/pretrained/swarm50_nn_checkpt.pth --make_vid
```

Quadcopter Experiment
```
python evalOC.py --nt 50 --resume experiments/oc/pretrained/singlequad_nn_checkpt.pth --make_vid
```



#### Train your own neural networks

Corridor Experiment
```
python trainOC.py --data softcorridor --lr_freq 600 --lr 0.01 --niters 1800 --lr_decay 0.1 --prec single --nt 20 --nt_val 32 --nTh 2 --var0 1.0 --log_freq 10 --val_freq 25 --viz_freq 100 --m 32 --n_train 1024 --sample_freq 100 --alph 100.0,10000.0,300.0,0.02,0.02,0.02
```

2-agent Swap Experiment
```
python trainOC.py --data swap2 --lr_freq 1000 --lr 0.01 --niters 4000 --lr_decay 0.1 --prec single --nt 20 --nt_val 42 --nTh 2 --var0 1.0 --log_freq 10 --val_freq 50 --viz_freq 100 --m 16 --n_train 1024 --sample_freq 50  --approach ocflow  --gpu 0 --alph 300.0,1000000.0,100000.0,1,1,3
```

12-agent Swap Experiment
```
python trainOC.py --data swap12 --lr_freq 800 --lr 0.01 --niters 2400 --lr_decay 0.1 --prec single --nt 20 --nt_val 32 --nTh 2 --var0 1.0 --val_freq 50 --viz_freq 100 --m 32 --n_train 2048 --sample_freq 100  --approach ocflow --gpu 0 --alph 300.0,0.0,100000.0,5.0,2.0,5.0 
```

Swarm Experiment
```
python trainOC.py --data swarm50 --lr_freq 1200 --lr 0.001 --niters 6000 --lr_decay 0.5 --prec single --nt 26 --nt_val 50 --nTh 2 --var0 0.1 --val_freq 25 --viz_freq 500 --m 512 --n_train 1024 --sample_freq 25 --approach ocflow  --gpu 0 --alph 1800.0,10000000.0,25000.0,2,1,3 --new_alph 800,900.0,10000000.0,25000.0,0.0,0.0,0.0
```

Quacopter Experiment
```
python3 trainOC.py --data singlequad --lr_freq 1200 --lr 0.01 --niters 6000 --lr_decay 0.5 --prec single --nt 26 --nt_val 50 --nTh 2 --var0 0.1 --val_freq 50 --viz_freq 100 --m 128 --n_train 1024  --sample_freq 25  --alph 5000.0,0.0,0.0,0.1,0.0,0.0
```

### Baseline Examples

#### Pretrained baseline examples

Corridor Baseline presented in publication

```
python baseline2D.py --data softcorridor --nt 50 --alph 100.0,10000.0,300.0 --resume experiments/oc/pretrained/softcorridor_baseline_checkpt.pth
```

Baseline and Neural Network comparisons presented in publications
```
python compareCorridor.py

python compareQuad.py
```

#### Train your own baseline examples
```
python baseline2D.py --data softcorridor --nt 50 --alph 100.0,10000.0,300.0

python baselineQuad.py --data singlequad --nt 50 --alph 5000.0,0.0,0.0
```


## Acknowledgments

This material is in part based upon work supported by the US National Science Foundation Grant DMS-1751636, the US AFOSR Grants 20RT0237 and FA9550-18-1-0167, AFOSR MURI FA9550-18-1-050, and ONR Grant No. N00014-18-1- 2527. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the funding agencies.





# State Advantage Weighting for Offline Reinforcement Learning (SAW)
Code for State Advantage Weighting for Offline RL (ICLR 2023 tiny paper, NeurIPS 2022 Offline RL workshop)

In this repo, we provide a super-clean pytorch implementation for SAW, which leverages state advantage $A(s,s^\prime) = Q(s,s^\prime) - V(s)$ for policy learning. We learn a policy conditioned on the next state, i.e., $\pi(s,s^\prime)$. It indicates what action an agent needs to take such that it can move from state $s$ to next state $s^\prime$.

## How to use
Run SAW by calling:
```
python main.py --env hopper-medium-v2 --expectile 0.7 --temperature 3.0
```

## Citation
If you use our code in your research, please consider cite:
```
@inproceedings{lyu2023state,
  title={State Advantage Weighting for Offline {RL}},
  author={Jiafei Lyu and Aicheng Gong and Le Wan and Zongqing Lu and Xiu Li},
  year={2023},
  booktitle={International Conference on Learning Representation tiny paper},
  url={https://openreview.net/forum?id=PjypHLTo29v}
}
```

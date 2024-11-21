# *DynSyn*: Dynamical Synergies for Efficient Generalization
## Requirement
gymnasium==0.29.1<br>
stable-baselines3==2.3.2<br>
sb3_contrib==2.3.0<br>
mujoco==3.1.2<br>

## Training Command

```
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python SB3-Scripts/train.py -f configs/locomotion-0.json
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python SB3-Scripts/train.py -f configs/manipulation-0.json
```

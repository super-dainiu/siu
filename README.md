# Siu: Solver integration utils for Colossal-AI

## Overview
This repo is somehow the minimal implementation of Colossal-AI FX. Features include:
- Robust Control-flow Tracing / Recompile
- Robust Activation Checkpoint Tracing / CodeGen
- ``MetaTensorMode``
- Shape Inference Accross Device and Unified ``MetaInfo``
- (TODO) Minimal Profiling for Solvers

## Install
```bash
git clone https://github.com/super-dainiu/siu.git
cd siu
pip install -r requirements.txt
pip install -e .
```

## Quickstart
### siu.FX

![image](https://user-images.githubusercontent.com/78588128/211300536-bf78bda4-1ec3-4b96-8f00-e067e5c6f343.png)

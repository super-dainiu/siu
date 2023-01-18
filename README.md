# Siu: Solver integration utils for Colossal-AI

# Overview
This repo is somehow the minimal implementation of Colossal-AI FX. Features include:
- Robust Control-flow Tracing / Recompile
- Robust Activation Checkpoint Tracing / CodeGen
- Support ``MetaTensorMode``
- Shape Inference Across Device and Unified ``MetaInfo``
- (TODO) Minimal Bias-Addition Split
- (TODO) Minimal Profiling for Solvers

# Install
```bash
git clone https://github.com/super-dainiu/siu.git
cd siu
pip install -r requirements.txt
pip install -e .
```

# Quickstart
## siu.FX
**Reference:**

  https://pytorch.org/docs/stable/fx.html [[paper](https://arxiv.org/pdf/2112.08429)]
  

torch.FX is a toolkit for developers to use to transform nn.Module instances. FX consists of three main components: a symbolic tracer, an intermediate representation, and Python code generation. FX.Tracer hacks _\_\_torch_function\_\__ and use an Proxy object to propagate through any forward function of torch.nn.Module.
![image](https://user-images.githubusercontent.com/78588128/212531495-bbb934dd-dbbb-4578-8869-6171973f7dd8.png)

### siu.fx.symbolic_trace()
A drawback of the original torch.FX implementation is that it is poor at handling control flow. All control-flow is not PyTorch native operands, and requires actual instances specify the branches to execute on. For example,

```python
class MyModule(nn.Module):
    def forward(self, x):
        if x.dim() == 3:
            return x * 2 + 1
        else:
            return x - 5
```

The above function has the computation graph of

![image](https://user-images.githubusercontent.com/78588128/212532631-dba30734-577b-4418-8dc9-004d7983abc5.png)

However, since Proxy does not have concrete data, applying ``x.dim()`` will return nothing. In the context of auto-parallel system, at least the control-flow dependencies for tensor shape should be removed, since any searched strategy could only auto-parallize a specific computation graph with same tensor shape. It is native to attach concrete data onto Proxy, and propagate them through control-flow.

![image](https://user-images.githubusercontent.com/78588128/212533403-1b620986-1c3a-420a-87c6-d08c9702135d.png)


With ``MetaTensor``, the computation during shape propagation can be virtualize. This speeds up tracing by avoiding allocating actual memory on devices.

#### Remarks
There is no free lunch for PyTorch to unify all operands in both its own repo and other repos in its eco-system. For example, the einops library currently has no intention to support torch.FX (See https://github.com/arogozhnikov/einops/issues/188). To support different PyTorch-based libraries without modifying source code, good practices can be to allow users register their implementation to substitute the functions not supported by torch.FX, or to avoid entering incompatible submodules.

### siu.fx.passes.ShapeProp
``ShapeProp`` is another important feature for Colossal-AI's auto-parallel system. Both Tensor Parallel and Activation Checkpoint solvers need to know the shape information ahead of time. Unlike PyTorch's implementation, this ShapeProp can be executed under MetaTensorMode. With this, all the preparation for auto-parallel solvers can be done in milliseconds.

![image](https://user-images.githubusercontent.com/78588128/211300536-bf78bda4-1ec3-4b96-8f00-e067e5c6f343.png)

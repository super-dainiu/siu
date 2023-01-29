# Siu: Solver integration utils for Colossal-AI

# Overview
This repo is somehow the minimal implementation of Colossal-AI FX. Features include:
- symbolic_trace()
  - Robust Control-flow Tracing / Recompile
  - Robust Activation Checkpoint Tracing / CodeGen
  - Minimal Bias-Addition Split
- symbolic_profile()
  - Support ``MetaTensorMode``
  - Shape Inference Across Device and Unified ``MetaInfo``
  - Ideal Flop Counter https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505

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
  

torch.FX is a toolkit for developers to use to transform nn.Module instances. FX consists of three main components: a symbolic tracer, an intermediate representation, and Python code generation. FX.Tracer hacks _\_\_torch_function\_\__ and use a Proxy object to propagate through any forward function of torch.nn.Module.
![image](https://user-images.githubusercontent.com/78588128/212531495-bbb934dd-dbbb-4578-8869-6171973f7dd8.png)
ColossalAI FX is modified from torch.FX, with the extra capability of ahead-of-time profiling enabled by the subclass of ``MetaTensor``.

### siu.fx.symbolic_trace()
A drawback of the original torch.FX implementation is that it is poor at handling control flow. All control flow is not PyTorch native operands and requires actual instances that specify the branches to execute on. For example,

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

However, since Proxy does not have concrete data, applying ``x.dim()`` will return nothing. In the context of the auto-parallel system, at least the control-flow dependencies for tensor shape should be removed, since any searched strategy could only auto-parallelize a specific computation graph with the same tensor shape. It is native to attach concrete data onto a Proxy, and propagate them through control flow.

![image](https://user-images.githubusercontent.com/78588128/212533403-1b620986-1c3a-420a-87c6-d08c9702135d.png)


With ``MetaTensor``, the computation during shape propagation can be virtualized. This speeds up tracing by avoiding allocating actual memory on devices.

#### Remarks
There is no free lunch for PyTorch to unify all operands in both its repo and other repos in its eco-system. For example, the einops library currently has no intention to support torch.FX (See https://github.com/arogozhnikov/einops/issues/188). To support different PyTorch-based libraries without modifying source code, good practices can be to allow users to register their implementation to substitute the functions not supported by torch.FX, or to avoid entering incompatible submodules.

### siu.fx.symbolic_profile
#### ShapeProp
``ShapeProp`` is another important feature of Colossal-AI's auto-parallel system. Both Tensor Parallel and Activation Checkpoint solvers need to know the shape information ahead of time. Unlike PyTorch's implementation, this ``ShapeProp`` can be executed under MetaTensorMode. With this, all the preparation for auto-parallel solvers can be done in milliseconds.

If a ``Graph`` is modified with some non-PyTorch functions, such as fused operands, you can register the shape propagation rule with the decorator.

```python
@register_shape_impl(fuse_conv_bn)
def fuse_conv_bn_shape_impl(*args, **kwargs):
     # do something here
     return torch.empty(output_shape, device=output_device)
```

An important notice is that ``ShapeProp`` will attach additional information to the graph, which will be exactly the input of ``GraphProfile``.

#### GraphProfile
``GraphProfile`` executes at the node level, and profiles both forward and backward within one node. However, the drawbacks of FX is that not every ``call_function`` saves its input for backward, and different tensor that flows within one FX.Graph can actually have the same layout. This raises problems for fine-grained profiling.

![image](https://user-images.githubusercontent.com/78588128/215312957-7eb6cbc3-61b2-49cf-95a4-6b859149eb8d.png)

To address this problem, I came up with a simulated environment enabled by ``torch.autograd.graph.saved_tensor_hooks`` and fake ``data_ptr``.
```python
class sim_env(saved_tensors_hooks):

    def __init__(self):
        super().__init__(self.pack_hook, self.unpack_hook)
        self.cache = {}

    def pack_hook(self, tensor: torch.Tensor):
        self.cache[tensor.data_ptr()] = tensor.unwrap() if hasattr(tensor, 'unwrap') else tensor
        return tensor

    def unpack_hook(self, tensor):
        return tensor
```
The ``cache`` variable will keep track of all saved tensors with a unique identifier.

![image](https://user-images.githubusercontent.com/78588128/211300536-bf78bda4-1ec3-4b96-8f00-e067e5c6f343.png)

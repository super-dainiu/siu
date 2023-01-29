import torch
import torch.distributed as dist

aten = torch.ops.aten

__all__ = [
    "_TorchFactoryMethod", "_TorchOverrideableFactoryMethod", "_TorchNonOverrideableFactoryMethod",
    "_TensorPropertyMethod", "_DistCommMethod", "_AliasATen", "_InplaceATen"
]

_TorchOverrideableFactoryMethod = [
    "empty",
    "eye",
    "full",
    "ones",
    "rand",
    "randn",
    "zeros",
]

_TorchNonOverrideableFactoryMethod = [
    "arange",
    "finfo",
    "linspace",
    "logspace",
    "randint",
    "randperm",
    "tensor",
]

_TorchFactoryMethod = _TorchOverrideableFactoryMethod + _TorchNonOverrideableFactoryMethod

_TensorPropertyMethod = ["dtype", "shape", "device", "requires_grad", "grad", "grad_fn", "data"]

_DistCommMethod = [
    "all_gather",
    "all_reduce",
    "all_to_all",
    "broadcast",
    "gather",
    "reduce",
    "reduce_scatter",
    "scatter",
]

_AliasATen = [
    aten.detach.default,
    aten.detach_.default,
    aten.t.default,
    aten.transpose.int,
    aten.view.default,
    aten._unsafe_view.default,
    aten._reshape_alias.default,
]

_InplaceATen = [
    aten.add_.Tensor,
    aten.add_.Scalar,
    aten.sub_.Tensor,
    aten.sub_.Scalar,
    aten.mul_.Tensor,
    aten.mul_.Scalar,
    aten.div_.Tensor,
    aten.div_.Scalar,
    aten.pow_.Tensor,
    aten.pow_.Scalar,
]

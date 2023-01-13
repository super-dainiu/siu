import enum
import functools
import inspect
import operator
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from torch.fx import Graph, Node, Proxy, Tracer
from torch.utils._pytree import tree_map

from siu._subclasses import MetaTensor, _TorchFactoryMethod, _TensorPropertyMethod
from .profiler_util import MetaInfo
from colossalai.fx import ColoGraphModule

Target = Union[Callable[..., Any], str]
Argument = Optional[Union[Tuple[Any, ...],    # actually Argument, but mypy can't represent recursive types
                          List[Any],    # actually Argument
                          Dict[str, Any],    # actually Argument
                          slice,    # Slice[Argument, Argument, Argument], but slice is not a templated type in typing
                          'Node',]]

def _truncate_suffix(s: str):
    import re
    return re.sub(r'_\d+$', '', s)


def _normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def _default_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class ColoProxy(Proxy):

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta_data = data

    @property
    def meta_data(self):
        return self._meta_data

    @meta_data.setter
    def meta_data(self, args):
        wrap_fn = lambda x: MetaTensor(x) if isinstance(x, torch.Tensor) else x
        self._meta_data = tree_map(wrap_fn, args)

    @classmethod
    def __torch_function__(cls, orig_method, types, args=(), kwargs=None):
        proxy = cls.from_torch_proxy(super().__torch_function__(orig_method, types, args, kwargs))
        unwrap_fn = lambda p: p.meta_data if isinstance(p, ColoProxy) else p
        kwargs = {} if kwargs is None else kwargs
        if proxy.meta_data is None:
            proxy.meta_data = orig_method(*tree_map(unwrap_fn, args), **tree_map(unwrap_fn, kwargs))
        return proxy

    @classmethod
    def from_torch_proxy(cls, proxy: Proxy):
        return cls(proxy.node, proxy.tracer)

    def __repr__(self):
        return f"ColoProxy({self.node.name}, meta_data={self.meta_data})"

    def __len__(self):
        return len(self.meta_data)

    def __int__(self):
        return int(self.meta_data)

    def __index__(self):
        try:
            return int(self.meta_data)
        except:
            return torch.zeros(self.meta_data.shape, dtype=torch.bool).numpy().__index__()

    def __float__(self):
        return float(self.meta_data)

    def __bool__(self):
        return self.meta_data

    def __getattr__(self, k):
        return ColoAttribute(self, k, getattr(self._meta_data, k, None))

    def __setitem__(self, key, value):
        proxy = self.tracer.create_proxy('call_function', operator.setitem, (self, key, value), {})
        proxy.meta_data = self._meta_data
        return proxy

    def __contains__(self, key):
        if self.node.op == "placeholder":
            # this is used to handle like
            # if x in kwargs
            # we don't handle this case for now
            return False
        return super().__contains__(key)

    def __isinstancecheck__(self, type):
        return isinstance(self.meta_data, type)

    @property
    def shape(self):
        return self.meta_data.shape

    @property
    def ndim(self):
        return self.meta_data.ndim

    @property
    def device(self):
        proxy = self.tracer.create_proxy('call_function', getattr, (self, 'device'), {})
        proxy.meta_data = self.meta_data.device
        return proxy

    @property
    def dtype(self):
        proxy = self.tracer.create_proxy('call_function', getattr, (self, 'dtype'), {})
        proxy.meta_data = self.meta_data.dtype
        return proxy

    def to(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', 'to', (self, *args), {**kwargs})

    def cpu(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', 'cpu', (self, *args), {**kwargs})

    def cuda(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', 'cuda', (self, *args), {**kwargs})


class ColoAttribute(ColoProxy):

    def __init__(self, root, attr: str, data=None):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._meta_data = data
        self._node: Optional[Node] = None

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy('call_function', getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy('call_method', self.attr, (self.root,) + args, kwargs)

    def __repr__(self):
        return f"ColoAttribute({self.node.name}, attr={self.attr})"


class ColoTracer(Tracer):

    def __init__(self, trace_act_ckpt: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_module_getattr = False
        self.proxy_buffer_attributes = True

        # whether the tracer will record the usage of torch.utils.checkpoint
        self.trace_act_ckpt = trace_act_ckpt
        # whether the current tracing occurs within the activation checkpoint functions
        self.ckpt_regions = []
        self.ckpt_idx = 0

    def proxy(self, node: Node) -> 'ColoProxy':
        return ColoProxy(node, self)

    def create_proxy(self,
                     kind: str,
                     target: Target,
                     args: Tuple[Any, ...],
                     kwargs: Dict[str, Any],
                     name: Optional[str] = None,
                     type_expr: Optional[Any] = None,
                     proxy_factory_fn: Callable[[Node], 'Proxy'] = None):

        proxy: ColoProxy = super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)
        unwrap_fn = lambda p: p.meta_data if isinstance(p, ColoProxy) else p
        if kind == 'placeholder':
            proxy.meta_data = self.meta_args[target] if target in self.meta_args else self.concrete_args.get(
                _truncate_suffix(target), None)
        elif kind == 'get_attr':
            self.disable_module_getattr = True
            try:
                attr_itr = self.root
                atoms = target.split(".")
                for atom in atoms:
                    attr_itr = getattr(attr_itr, atom)
                proxy.meta_data = attr_itr
            finally:
                self.disable_module_getattr = False
        elif kind == 'call_function':
            proxy.meta_data = target(*tree_map(unwrap_fn, args), **tree_map(unwrap_fn, kwargs))
        elif kind == 'call_method':
            self.disable_module_getattr = True
            try:
                if target == '__call__':
                    proxy.meta_data = unwrap_fn(args[0])(*tree_map(unwrap_fn, args[1:]), **tree_map(unwrap_fn, kwargs))
                else:
                    if target not in _TensorPropertyMethod:
                        proxy._meta_data = getattr(unwrap_fn(args[0]), target)(*tree_map(unwrap_fn, args[1:]),
                                                                               **tree_map(unwrap_fn, kwargs))
            finally:
                self.disable_module_getattr = False
        elif kind == 'call_module':
            mod = self.root.get_submodule(target)
            self.disable_module_getattr = True
            try:
                proxy.meta_data = mod.forward(*tree_map(unwrap_fn, args), **tree_map(unwrap_fn, kwargs))
            finally:
                self.disable_module_getattr = False
        return proxy

    def create_node(self, *args, **kwargs) -> Node:
        node = super().create_node(*args, **kwargs)
        n_info = MetaInfo(node, to_recompute=tuple(self.ckpt_regions))
        return node

    def trace(self,
              root: torch.nn.Module,
              concrete_args: Optional[Dict[str, torch.Tensor]] = {},
              meta_args: Optional[Dict[str, torch.Tensor]] = {}) -> Graph:

        # check concrete and meta args have valid names
        sig = inspect.signature(root.forward)
        sig_names = set(sig.parameters.keys())
        meta_arg_names = set(meta_args.keys())
        concrete_arg_names = set(concrete_args.keys())

        # update concrete args with default values
        for k, v in sig.parameters.items():
            if k in sig_names - meta_arg_names and \
                    k not in concrete_args and \
                    v.default is not inspect.Parameter.empty:
                concrete_args[k] = v.default

        def _check_arg_name_valid(names: Iterable[str]):
            for name in names:
                if name not in sig_names:
                    raise ValueError(f"Argument {name} is not in the signature of {root.__class__.__name__}.forward")

        _check_arg_name_valid(meta_arg_names)
        _check_arg_name_valid(concrete_arg_names)

        self.concrete_args = concrete_args
        self.meta_args = meta_args

        with _TorchTensorOverride(self), self.trace_activation_checkpoint(enabled=self.trace_act_ckpt):
            self.graph = super().trace(root, concrete_args=concrete_args)
        self.graph.lint()
        return self.graph

    @contextmanager
    def trace_activation_checkpoint(self, enabled: bool):
        if enabled:
            orig_ckpt_func = torch.utils.checkpoint.CheckpointFunction

            class PatchedCheckpointFunction(torch.autograd.Function):

                @staticmethod
                def forward(ctx, run_function, preserve_rng_state, *args):
                    # signal that the current tracing occurs within activaton checkpoint part
                    self.ckpt_regions.append(self.ckpt_idx)
                    out = run_function(*args)
                    self.ckpt_idx = self.ckpt_regions.pop(-1) + 1
                    return out

                @staticmethod
                def backward(ctx: Any, *grad_outputs: Any) -> Any:
                    raise NotImplementedError(
                        "We do not implement the backward pass as we only trace the forward pass.")

            # override the checkpoint function
            torch.utils.checkpoint.CheckpointFunction = PatchedCheckpointFunction
        yield

        if enabled:
            # recover the checkpoint function upon exit
            torch.utils.checkpoint.CheckpointFunction = orig_ckpt_func

    def _post_check(self, non_concrete_arg_names: Set[str]):
        # This is necessary because concrete args are added as input to the traced module since
        # https://github.com/pytorch/pytorch/pull/55888.
        for node in self.graph.nodes:
            if node.op == "placeholder":
                # Removing default values for inputs as the forward pass will fail with them.
                if node.target in non_concrete_arg_names:
                    node.args = ()
                    # Without this, torch.jit.script fails because the inputs type is Optional[torch.Tensor].
                    # It cannot infer on the attributes and methods the input should have, and fails.
                    node.type = torch.Tensor
                # It is a concrete arg so it is not used and should be removed.
                else:
                    if hasattr(torch.fx._symbolic_trace, "_assert_is_none"):
                        # Newer versions of torch.fx emit an assert statement
                        # for concrete arguments; delete those before we delete
                        # the concrete arg.
                        to_delete = []
                        for user in node.users:
                            if user.target == torch.fx._symbolic_trace._assert_is_none:
                                to_delete.append(user)
                        for user in to_delete:
                            self.graph.erase_node(user)

                    self.graph.erase_node(node)

            # TODO: solves GraphModule creation.
            # Without this, return type annotation "Tuple" is causing code execution failure.
            if node.op == "output":
                node.type = None
            self.graph.lint()

    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, "disable_module_getattr", False):
            return attr_val

        def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if 'proxy_factory_fn' in inspect.signature(self.create_proxy).parameters:
                            kwargs['proxy_factory_fn'] = (None if not self.param_shapes_constant else
                                                          lambda node: ColoProxy(self, node, n, attr_val))
                        val_proxy = self.create_proxy('get_attr', n, (), {}, **kwargs)    # type: ignore[arg-type]
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None

        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_buffers(), parameter_proxy_cache)
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy

        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(attr_val, self.root.named_parameters(),
                                                             parameter_proxy_cache)
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        return attr_val


def symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = {},
    meta_args: Optional[Dict[str, Any]] = {},
    trace_act_ckpt=False,
) -> ColoGraphModule:
    if meta_args:
        device = _default_device()
        wrap_fn = lambda elem: MetaTensor(elem, device=device) if isinstance(elem, torch.Tensor) else elem
        graph = ColoTracer(trace_act_ckpt=trace_act_ckpt).trace(root.to(device),
                                                                concrete_args=concrete_args,
                                                                meta_args=tree_map(wrap_fn, meta_args))
    else:
        graph = Tracer().trace(root, concrete_args=concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return ColoGraphModule(root, graph, name)


class _TorchTensorOverride(object):

    def __init__(self, tracer: Tracer):
        self.overrides = {}
        self.tracer = tracer

    def __enter__(self):

        def wrap_tensor_method(target):

            @functools.wraps(target)
            def wrapper(*args, **kwargs):
                is_proxy = any(isinstance(p, ColoProxy) for p in args) | any(
                    isinstance(p, ColoProxy) for p in kwargs.values())
                if is_proxy:
                    # if the arg is a proxy, then need to record this function called on this proxy
                    # e.g. torch.ones(size) where size is an input proxy
                    self.tracer.disable_module_getattr = True
                    try:
                        proxy = self.tracer.create_proxy('call_function', target, args, kwargs)
                    finally:
                        self.tracer.disable_module_getattr = False
                    return proxy
                else:
                    return target(*args, **kwargs)

            return wrapper, target

        self.overrides = {
            target: wrap_tensor_method(getattr(torch, target))
            for target in _TorchFactoryMethod
            if callable(getattr(torch, target))
        }
        for name, (wrapper, orig) in self.overrides.items():
            setattr(torch, name, wrapper)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, (wrapper, orig) in self.overrides.items():
            setattr(torch, name, orig)
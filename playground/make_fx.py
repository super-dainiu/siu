import torch
from torch.fx import Graph, Node
from torch.utils._pytree import tree_map


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def make_fx(module: torch.nn.Module, *args, device=None, **kwargs) -> Graph:
    """Trace forward and backward graph with MetaTensor

    Args:
        module (torch.nn.Module): The target module for tracing.

    Returns:
        graph (torch.fx.Graph): The computation graph.

    Usage:
        >>> import torchvision.models as tm
        >>> model = tm.alexnet()
        >>> graph = meta_trace(model, torch.rand(1000, 3, 224, 224))
        >>> graph.print_tabular()
    """
    graph = Graph()
    namespace = graph._graph_namespace

    def is_autogradable(r):
        return isinstance(r, torch.Tensor) and r.is_floating_point()

    class MetaProxy(torch.Tensor):
        """
        A wrapping tensor that hacks `torch.autograd` without patching more `torch.ops.aten` ops.
        """

        _tensor: torch.Tensor
        _node: Node

        __slots__ = ['_tensor', '_node']

        @staticmethod
        def __new__(cls, tensor, device=None, placeholder=False, name=None):
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                tensor.size(),
                strides=tensor.stride(),
                storage_offset=tensor.storage_offset(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                device=device if device is not None else tensor.device,
                requires_grad=tensor.requires_grad)    # deceive the frontend for aten selections
            r._tensor = tensor
            if placeholder:
                if name is None:
                    name = 'input'
                r._node = graph.create_node('placeholder',
                                            'placeholder', (graph._root,),
                                            name=namespace.create_name(name, tensor))
            # ...the real tensor is held as an element on the tensor.
            if not r._tensor.is_meta:
                r._tensor = r._tensor.to(torch.device('meta'))
            return r

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

            def unwrap(x):
                nonlocal device
                if isinstance(x, MetaProxy):
                    device = x.device
                    x = x._tensor
                elif isinstance(x, torch.Tensor):
                    device = x.device
                    x = x.to(torch.device('meta'))
                return x

            def get_node(x):
                if isinstance(x, torch.Tensor) and not hasattr(x, '_node'):
                    x = MetaProxy(x, placeholder=True, name='weight')
                return x if not hasattr(x, '_node') else x._node

            args_node = tree_map(get_node, args)
            kwargs_node = tree_map(get_node, kwargs)
            node = graph.create_node('call_function', func, args_node, kwargs_node)

            if 'device' in kwargs:
                device = kwargs['device']
                kwargs['device'] = torch.device('meta')

            args = tree_map(unwrap, args)
            kwargs = tree_map(unwrap, kwargs)

            # run aten for backend=CPU but actually on backend=Meta
            out = func(*args, **kwargs)

            # Now, we want to continue propagating this tensor, so we rewrap Tensors in
            # our custom tensor subclass
            def wrap(x):
                if isinstance(x, torch.Tensor):
                    nonlocal device
                    if not x.is_meta:
                        x = x.to(torch.device('meta'))
                return MetaProxy(x, device=device) if isinstance(x, torch.Tensor) and not hasattr(x, '_tensor') else x

            def set_node(x: MetaProxy):
                x._node = node

            out = tree_map(wrap, out)
            tree_map(set_node, out)

            return out

    def wrap(x):
        return MetaProxy(x, device=device, placeholder=True) if isinstance(x, torch.Tensor) else x

    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)

    rst = module(*args, **kwargs)
    rst = tuple(r for r in normalize_tuple(rst) if is_autogradable(r) and r.requires_grad)

    if rst:
        grad = [torch.zeros_like(t) for t in rst]
        torch.autograd.backward(
            rst,
            grad,
            retain_graph=True,
        )
    return graph

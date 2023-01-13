"""``torch.fx.ShapeProp``, but with ``MetaTensor``"""

from typing import Any, Dict, Tuple, Union, Callable

import torch
from torch.utils._pytree import tree_map
import torch.fx

from siu._subclasses import MetaTensor, MetaTensorMode
from siu.fx.profiler_util import MetaInfo

Target = Union[Callable[..., Any], str]


def _normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def register_shape_impl(func):
    def wrapper(impl):
        ShapeProp._custom_dispatch_func[func] = impl
        return impl

    return wrapper

class ShapeProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and record the meta data of the result
    into the corresponding node.

    Usage:
        BATCH_SIZE = 2
        DIM_IN = 4
        DIM_HIDDEN = 16
        DIM_OUT = 16
        model = torch.nn.Sequential(
            torch.nn.Linear(DIM_IN, DIM_HIDDEN),
            torch.nn.Linear(DIM_HIDDEN, DIM_OUT),
            )
        input_sample = torch.rand(BATCH_SIZE, DIM_IN)
        gm = colossalai.fx.symbolic_trace(model)
        interp = ShapeProp(gm)
        interp.propagate(input_sample)

    Args:
        module (GraphModule): The module to be executed

    Hints:
        If you want to add a new shape propagation rule, you can do so by
        adding a new method to this class with the ``@register_shape_impl``
        decorator. The method should take (*args, **kwargs) instance as its
        input and generate output.

        For example, if you want to add a shape propagation rule for 
        ``torch.nn.functional.linear``, you can do so by adding a new method
        to this class with the ``@register_shape_impl`` decorator (Since the
        ``MetaTensorMode`` is compatible with ``torch.nn.functional.linear``,
        in practice you don't have to do as follows):

        >>> @register_shape_impl(torch.nn.functional.linear)
        >>> def linear_shape_impl(*args, **kwargs):
        >>>     # do something here
        >>>     return torch.empty(output_shape, device=output_device)
    """
    _custom_dispatch_func = {}
    _mode = MetaTensorMode()

    def run_node(self, n: torch.fx.Node) -> Any:
        """
        Run a specific node ``n`` and return the result. Attach 'data' to ``n``.

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        r = super().run_node(n)
        unwrap_fn = lambda elem: elem._tensor if isinstance(elem, MetaTensor) else elem
        n_info = MetaInfo(n)
        n_info.data = tree_map(unwrap_fn, _normalize_tuple(r))
        if n.op == 'call_module':
            submod = self.fetch_attr(n.target)
            n_info.parameters = {k: v.to(torch.device('meta')) for k, v in submod.named_parameters()}
            n_info.buffers = {k: v.to(torch.device('meta')) for k, v in submod.named_buffers()}

        # TODO(you): Remove this once SPMD Solver is refactored.
        n._meta_data = tree_map(unwrap_fn, _normalize_tuple(r))
        return r

    def call_function(self, target: 'Target', args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if target in self._custom_dispatch_func:
            return self._custom_dispatch_func[target](self, args, kwargs)
        return super().call_function(target, args, kwargs)

    def propagate(self, *args, device=None):
        """
        Run `module` via interpretation and return the result and record the
        shape of each node.
        Args:
            *args (Tensor): the sample input.
        Returns:
            Any: The value returned from executing the Module
        """
        wrap_fn = lambda elem: MetaTensor(elem, device=device)
        with self._mode:
            return super().run(*tree_map(wrap_fn, args))


def shape_prop_pass(gm: torch.fx.GraphModule, *args, device=None):
    ShapeProp(gm).propagate(*args, device=device)
    return gm

from dataclasses import dataclass
from typing import Any, Iterable, Tuple
import queue
from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals_eps = []
    for i in range(len(vals)):
        vals_eps.append(vals[i])
        if i == arg:
            vals_eps[i] += epsilon

    return (f(*vals_eps) - f(*vals)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    res = []
    visited = set()

    def dfs(node: Variable) -> None:
        if node.is_constant():
            return
        if node.unique_id in visited:
            return
        visited.add(node.unique_id)
        for par in node.parents:
            dfs(par)

        res.append(node)

    dfs(variable)
    res = res[::-1]
    return res


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    accum_derivatives = dict()
    accum_derivatives[variable.unique_id] = deriv
    for var in topological_sort(variable):
        if var.unique_id in accum_derivatives.keys():
            d_output = accum_derivatives[var.unique_id]
        else:
            d_output = 0.0
        if var.is_leaf():
            var.accumulate_derivative(d_output)
        else:
            for parent, d_parent in var.chain_rule(d_output):
                if parent.unique_id in accum_derivatives.keys():
                    accum_derivatives[parent.unique_id] += d_parent
                else:
                    accum_derivatives[parent.unique_id] = d_parent


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values

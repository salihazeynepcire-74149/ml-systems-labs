"""
Functional Reverse-Mode Automatic Differentiation Core
Student: Saliha Zeynep Cire (Student ID: 74149)
Course: Machine Learning Systems (Lab 05)
Institution: Vistula University
"""

from __future__ import annotations
import math
from typing import Set, Tuple, Callable, Optional, Union, List


class Variable:
    """
    Differentiable scalar variable representing a computation graph node.
    Maintains the primal scalar, adjoint derivative accumulation, dependencies,
    and a local gradient propagation rule.
    """
    __slots__ = ('x', 'grad', '_deps', '_vjp', 'op_code', 'tag')

    def __init__(
        self,
        x: Union[float, int],
        _deps: Tuple[Variable, ...] = (),
        op_code: str = '',
        tag: Optional[str] = None
    ) -> None:
        self.x: float = float(x)
        self.grad: float = 0.0
        self._deps: Set[Variable] = set(_deps)
        self._vjp: Callable[[], None] = lambda: None
        self.op_code: str = op_code
        self.tag: Optional[str] = tag

    def __repr__(self) -> str:
        ident = f" [{self.tag}]" if self.tag else ""
        return f"Variable(x={self.x:.4f}, grad={self.grad:.4f}{ident})"

    def __add__(self, other: Union[Variable, float, int]) -> Variable:
        rhs = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.x + rhs.x, (self, rhs), '+')

        def _vjp_add() -> None:
            self.grad += 1.0 * out.grad
            rhs.grad += 1.0 * out.grad

        out._vjp = _vjp_add
        return out

    def __radd__(self, other: Union[float, int]) -> Variable:
        return self.__add__(other)

    def __mul__(self, other: Union[Variable, float, int]) -> Variable:
        rhs = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.x * rhs.x, (self, rhs), '*')

        def _vjp_mul() -> None:
            self.grad += rhs.x * out.grad
            rhs.grad += self.x * out.grad

        out._vjp = _vjp_mul
        return out

    def __rmul__(self, other: Union[float, int]) -> Variable:
        return self.__mul__(other)

    def __pow__(self, exponent: Union[float, int]) -> Variable:
        assert isinstance(exponent, (int, float)), "Power exponent must be float or integer"
        out = Variable(self.x ** exponent, (self,), f'^{exponent}')

        def _vjp_pow() -> None:
            self.grad += (exponent * (self.x ** (exponent - 1))) * out.grad

        out._vjp = _vjp_pow
        return out

    def relu(self) -> Variable:
        val = self.x if self.x > 0.0 else 0.0
        out = Variable(val, (self,), 'ReLU')

        def _vjp_relu() -> None:
            self.grad += (1.0 if self.x > 0.0 else 0.0) * out.grad

        out._vjp = _vjp_relu
        return out

    def sigmoid(self) -> Variable:
        if self.x >= 0.0:
            s = 1.0 / (1.0 + math.exp(-self.x))
        else:
            s = math.exp(self.x) / (1.0 + math.exp(self.x))
        out = Variable(s, (self,), 'Sigmoid')

        def _vjp_sigmoid() -> None:
            self.grad += (s * (1.0 - s)) * out.grad

        out._vjp = _vjp_sigmoid
        return out

    def log(self, eps: float = 1e-12) -> Variable:
        val = max(self.x, eps)
        out = Variable(math.log(val), (self,), 'Log')

        def _vjp_log() -> None:
            self.grad += (1.0 / val) * out.grad

        out._vjp = _vjp_log
        return out

    def __neg__(self) -> Variable:
        return self * -1.0

    def __sub__(self, other: Union[Variable, float, int]) -> Variable:
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> Variable:
        return Variable(other) + (-self)

    def __truediv__(self, other: Union[Variable, float, int]) -> Variable:
        rhs = other if isinstance(other, Variable) else Variable(other)
        return self * (rhs ** -1.0)

    def backward(self) -> None:
        """
        Executes reverse-mode automatic differentiation:
        - Topologically sorts dependencies using post-order depth-first traversal.
        - Sets root adjoint to 1.0.
        - Iteratively evaluates local VJPs in reverse topological order.
        - Accumulates adjoints in place using +=.
        """
        topo: List[Variable] = []
        visited: Set[Variable] = set()

        def dfs(node: Variable) -> None:
            if node not in visited:
                visited.add(node)
                for dep in node._deps:
                    dfs(dep)
                topo.append(node)

        dfs(self)
        self.grad = 1.0

        for n in reversed(topo):
            n._vjp()

"""
    pint.pint_eval
    ~~~~~~~~~~~~~~

    An expression evaluator to be used as a safe replacement for builtin eval.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
from __future__ import annotations

import operator
import token as tokenlib
from tokenize import TokenInfo

from typing import Any, Optional, Union

from .errors import DefinitionSyntaxError

# For controlling order of operations
_OP_PRIORITY = {
    "**": 3,
    "^": 3,
    "unary": 2,
    "*": 1,
    "": 1,  # operator for implicit ops
    "//": 1,
    "/": 1,
    "%": 1,
    "+": 0,
    "-": 0,
}


def _power(left: Any, right: Any) -> Any:
    from . import Quantity
    from .compat import is_duck_array

    if (
        isinstance(left, Quantity)
        and is_duck_array(left.magnitude)
        and left.dtype.kind not in "cf"
        and right < 0
    ):
        left = left.astype(float)

    return operator.pow(left, right)


import typing

UnaryOpT = typing.Callable[
    [
        Any,
    ],
    Any,
]
BinaryOpT = typing.Callable[[Any, Any], Any]

_UNARY_OPERATOR_MAP: dict[str, UnaryOpT] = {"+": lambda x: x, "-": lambda x: x * -1}

_BINARY_OPERATOR_MAP: dict[str, BinaryOpT] = {
    "**": _power,
    "*": operator.mul,
    "": operator.mul,  # operator for implicit ops
    "/": operator.truediv,
    "+": operator.add,
    "-": operator.sub,
    "%": operator.mod,
    "//": operator.floordiv,
}


class EvalTreeNode:
    """Single node within an evaluation tree

    left + operator + right --> binary op
    left + operator --> unary op
    left + right --> implicit op
    left --> single value
    """

    def __init__(
        self,
        left: Union[EvalTreeNode, TokenInfo],
        operator: Optional[TokenInfo] = None,
        right: Optional[EvalTreeNode] = None,
    ):
        self.left = left
        self.operator = operator
        self.right = right

    def to_string(self) -> str:
        # For debugging purposes
        if self.right:
            assert isinstance(self.left, EvalTreeNode), "self.left not EvalTreeNode (1)"
            comps = [self.left.to_string()]
            if self.operator:
                comps.append(self.operator.string)
            comps.append(self.right.to_string())
        elif self.operator:
            assert isinstance(self.left, EvalTreeNode), "self.left not EvalTreeNode (2)"
            comps = [self.operator.string, self.left.to_string()]
        else:
            assert isinstance(self.left, TokenInfo), "self.left not TokenInfo (1)"
            return self.left.string
        return "(%s)" % " ".join(comps)

    def evaluate(
        self,
        define_op: typing.Callable[
            [
                Any,
            ],
            Any,
        ],
        bin_op: Optional[dict[str, BinaryOpT]] = None,
        un_op: Optional[dict[str, UnaryOpT]] = None,
    ):
        """Evaluate node.

        Parameters
        ----------
        define_op : callable
            Translates tokens into objects.
        bin_op : dict or None, optional
             (Default value = _BINARY_OPERATOR_MAP)
        un_op : dict or None, optional
             (Default value = _UNARY_OPERATOR_MAP)

        Returns
        -------

        """

        bin_op = bin_op or _BINARY_OPERATOR_MAP
        un_op = un_op or _UNARY_OPERATOR_MAP

        if self.right:
            assert isinstance(self.left, EvalTreeNode), "self.left not EvalTreeNode (3)"
            # binary or implicit operator
            op_text = self.operator.string if self.operator else ""
            if op_text not in bin_op:
                raise DefinitionSyntaxError(f"missing binary operator '{op_text}'")

            return bin_op[op_text](
                self.left.evaluate(define_op, bin_op, un_op),
                self.right.evaluate(define_op, bin_op, un_op),
            )
        elif self.operator:
            assert isinstance(self.left, EvalTreeNode), "self.left not EvalTreeNode (4)"
            # unary operator
            op_text = self.operator.string
            if op_text not in un_op:
                raise DefinitionSyntaxError(f"missing unary operator '{op_text}'")
            return un_op[op_text](self.left.evaluate(define_op, bin_op, un_op))

        # single value
        return define_op(self.left)


from collections.abc import Iterable


def _build_eval_tree(
    tokens: list[TokenInfo],
    op_priority: dict[str, int],
    index: int = 0,
    depth: int = 0,
    prev_op: str = "<none>",
) -> tuple[EvalTreeNode, int]:
    """Build an evaluation tree from a set of tokens.

    Params:
    Index, depth, and prev_op used recursively, so don't touch.
    Tokens is an iterable of tokens from an expression to be evaluated.

    Transform the tokens from an expression into a recursive parse tree, following order
    of operations. Operations can include binary ops (3 + 4), implicit ops (3 kg), or
    unary ops (-1).

    General Strategy:
    1) Get left side of operator
    2) If no tokens left, return final result
    3) Get operator
    4) Use recursion to create tree starting at token on right side of operator (start at step #1)
    4.1) If recursive call encounters an operator with lower or equal priority to step #2, exit recursion
    5) Combine left side, operator, and right side into a new left side
    6) Go back to step #2

    Raises
    ------
    DefinitionSyntaxError
        If there is a syntax error.

    """

    result = None

    while True:
        current_token = tokens[index]
        token_type = current_token.type
        token_text = current_token.string

        if token_type == tokenlib.OP:
            if token_text == ")":
                if prev_op == "<none>":
                    raise DefinitionSyntaxError(
                        f"unopened parentheses in tokens: {current_token}"
                    )
                elif prev_op == "(":
                    # close parenthetical group
                    assert result is not None
                    return result, index
                else:
                    # parenthetical group ending, but we need to close sub-operations within group
                    assert result is not None
                    return result, index - 1
            elif token_text == "(":
                # gather parenthetical group
                right, index = _build_eval_tree(
                    tokens, op_priority, index + 1, 0, token_text
                )
                if not tokens[index][1] == ")":
                    raise DefinitionSyntaxError("weird exit from parentheses")
                if result:
                    # implicit op with a parenthetical group, i.e. "3 (kg ** 2)"
                    result = EvalTreeNode(left=result, right=right)
                else:
                    # get first token
                    result = right
            elif token_text in op_priority:
                if result:
                    # equal-priority operators are grouped in a left-to-right order,
                    # unless they're exponentiation, in which case they're grouped
                    # right-to-left this allows us to get the expected behavior for
                    # multiple exponents
                    #     (2^3^4)  --> (2^(3^4))
                    #     (2 * 3 / 4) --> ((2 * 3) / 4)
                    if op_priority[token_text] <= op_priority.get(
                        prev_op, -1
                    ) and token_text not in ("**", "^"):
                        # previous operator is higher priority, so end previous binary op
                        return result, index - 1
                    # get right side of binary op
                    right, index = _build_eval_tree(
                        tokens, op_priority, index + 1, depth + 1, token_text
                    )
                    result = EvalTreeNode(
                        left=result, operator=current_token, right=right
                    )
                else:
                    # unary operator
                    right, index = _build_eval_tree(
                        tokens, op_priority, index + 1, depth + 1, "unary"
                    )
                    result = EvalTreeNode(left=right, operator=current_token)
        elif token_type in (tokenlib.NUMBER, tokenlib.NAME):
            if result:
                # tokens with an implicit operation i.e. "1 kg"
                if op_priority[""] <= op_priority.get(prev_op, -1):
                    # previous operator is higher priority than implicit, so end
                    # previous binary op
                    return result, index - 1
                right, index = _build_eval_tree(
                    tokens, op_priority, index, depth + 1, ""
                )
                result = EvalTreeNode(left=result, right=right)
            else:
                # get first token
                result = EvalTreeNode(left=current_token)

        if tokens[index][0] == tokenlib.ENDMARKER:
            if prev_op == "(":
                raise DefinitionSyntaxError("unclosed parentheses in tokens")
            if depth > 0 or prev_op:
                # have to close recursion
                assert result is not None
                return result, index
            else:
                # recursion all closed, so just return the final result
                assert result is not None
                return result, -1

        if index + 1 >= len(tokens):
            # should hit ENDMARKER before this ever happens
            raise DefinitionSyntaxError("unexpected end to tokens")

        index += 1


def build_eval_tree(
    tokens: Iterable[TokenInfo],
    op_priority: Optional[dict[str, int]] = None,
) -> EvalTreeNode:
    """Build an evaluation tree from a set of tokens.

    Params:
    Index, depth, and prev_op used recursively, so don't touch.
    Tokens is an iterable of tokens from an expression to be evaluated.

    Transform the tokens from an expression into a recursive parse tree, following order
    of operations. Operations can include binary ops (3 + 4), implicit ops (3 kg), or
    unary ops (-1).

    General Strategy:
    1) Get left side of operator
    2) If no tokens left, return final result
    3) Get operator
    4) Use recursion to create tree starting at token on right side of operator (start at step #1)
    4.1) If recursive call encounters an operator with lower or equal priority to step #2, exit recursion
    5) Combine left side, operator, and right side into a new left side
    6) Go back to step #2

    Raises
    ------
    DefinitionSyntaxError
        If there is a syntax error.

    """

    if op_priority is None:
        op_priority = _OP_PRIORITY

    if not isinstance(tokens, list):
        # ensure tokens is list so we can access by index
        tokens = list(tokens)

    result, _ = _build_eval_tree(tokens, op_priority, 0, 0)

    return result

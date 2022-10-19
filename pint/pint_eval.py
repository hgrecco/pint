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
import tokenize

try:
    from uncertainties import ufloat
    HAS_UNCERTAINTIES = True
except ImportError:
    HAS_UNCERTAINTIES = False
    ufloat = None

from .errors import DefinitionSyntaxError

# For controlling order of operations
_OP_PRIORITY = {
    "±": 4,
    "+/-": 4,
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


def _ufloat(left, right):
    if HAS_UNCERTAINTIES:
        return ufloat(left, right)
    raise TypeError ('Could not import support for uncertainties')


def _power(left, right):
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


_BINARY_OPERATOR_MAP = {
    "±": _ufloat,
    "+/-": _ufloat,
    "**": _power,
    "*": operator.mul,
    "": operator.mul,  # operator for implicit ops
    "/": operator.truediv,
    "+": operator.add,
    "-": operator.sub,
    "%": operator.mod,
    "//": operator.floordiv,
}

_UNARY_OPERATOR_MAP = {"+": lambda x: x, "-": lambda x: x * -1}


class EvalTreeNode:
    """Single node within an evaluation tree

    left + operator + right --> binary op
    left + operator --> unary op
    left + right --> implicit op
    left --> single value
    """

    def __init__(self, left, operator=None, right=None):
        self.left = left
        self.operator = operator
        self.right = right

    def to_string(self):
        # For debugging purposes
        if self.right:
            comps = [self.left.to_string()]
            if self.operator:
                comps.append(self.operator[1])
            comps.append(self.right.to_string())
        elif self.operator:
            comps = [self.operator[1], self.left.to_string()]
        else:
            return self.left[1]
        return "(%s)" % " ".join(comps)

    def evaluate(self, define_op, bin_op=None, un_op=None):
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
            # binary or implicit operator
            op_text = self.operator[1] if self.operator else ""
            if op_text not in bin_op:
                raise DefinitionSyntaxError('missing binary operator "%s"' % op_text)
            left = self.left.evaluate(define_op, bin_op, un_op)
            return bin_op[op_text](left, self.right.evaluate(define_op, bin_op, un_op))
        elif self.operator:
            # unary operator
            op_text = self.operator[1]
            if op_text not in un_op:
                raise DefinitionSyntaxError('missing unary operator "%s"' % op_text)
            return un_op[op_text](self.left.evaluate(define_op, bin_op, un_op))
        else:
            # single value
            return define_op(self.left)


from typing import Iterable


def peek_exp_number(tokens, index):
    exp_number = None
    exp_number_end = index
    exp_is_negative = False
    if (
        index + 2 < len(tokens)
        and tokens[index + 1].string == "10"
        and tokens[index + 2].string in "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"
    ):
        if tokens[index + 2].string == "⁻":
            exp_is_negative = True
        for exp_number_end in range(index + 3, len(tokens)):
            if tokens[exp_number_end].string not in "⁰¹²³⁴⁵⁶⁷⁸⁹":
                break
        exp_number = "".join(
            [
                digit.string[0] - "⁰"
                for digit in tokens[index + exp_is_negative + 2 : exp_number_end]
            ]
        )
    else:
        if (
            index + 2 < len(tokens)
            and tokens[index + 1].string == "e"
            # No sign on the exponent, treat as +
            and tokens[index + 2].type == tokenlib.NUMBER
        ):
            # Don't know why tokenizer doesn't bundle all these numbers together
            for exp_number_end in range(index + 3, len(tokens)):
                if tokens[exp_number_end].type != tokenlib.NUMBER:
                    break
        elif (
            index + 3 < len(tokens)
            and tokens[index + 1].string == "e"
            and tokens[index + 2].string in ["+", "-"]
            and tokens[index + 3].type == tokenlib.NUMBER
        ):
            if tokens[index + 2].string == "-":
                exp_is_negative = True
            # Don't know why tokenizer doesn't bundle all these numbers together
            for exp_number_end in range(index + 4, len(tokens)):
                if tokens[exp_number_end].type != tokenlib.NUMBER:
                    break
        if exp_number_end > index:
            exp_number = "".join(
                [digit.string for digit in tokens[index + 3 : exp_number_end]]
            )
        else:
            return None, index
    exp_number = "1.0e" + ("-" if exp_is_negative else "") + exp_number
    assert exp_number_end != index
    return exp_number, exp_number_end


def finish_exp_number(tokens, exp_number, exp_number_end, plus_minus_op, left, right):
    exp_number_token = tokenize.TokenInfo(
        type=tokenlib.NUMBER,
        string=exp_number,
        start=(1, 0),
        end=(1, len(exp_number)),
        line=exp_number,
    )
    e_notation_operator = tokenize.TokenInfo(
        type=tokenlib.OP,
        string="*",
        start=(1, 0),
        end=(1, 1),
        line="*",
    )
    e_notation_scale, _ = build_eval_tree(
        [exp_number_token, tokens[-1]],
        None,
        0,
        0,
        tokens[exp_number_end].string,
    )
    scaled_left = EvalTreeNode(left, e_notation_operator, e_notation_scale)
    scaled_right = EvalTreeNode(right, e_notation_operator, e_notation_scale)
    result = EvalTreeNode(scaled_left, plus_minus_op, scaled_right)
    index = exp_number_end
    return result, index


def build_eval_tree(
    tokens: Iterable[tokenize.TokenInfo],
    op_priority=None,
    index=0,
    depth=0,
    prev_op=None,
) -> tuple[EvalTreeNode | None, int] | EvalTreeNode:
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

    """

    if op_priority is None:
        op_priority = _OP_PRIORITY

    if depth == 0 and prev_op is None:
        # ensure tokens is list so we can access by index
        tokens = list(tokens)

    result = None

    while True:
        current_token = tokens[index]
        token_type = current_token.type
        token_text = current_token.string

        if token_type == tokenlib.OP:
            if token_text == ")":
                if prev_op is None:
                    raise DefinitionSyntaxError(
                        "unopened parentheses in tokens: %s" % current_token
                    )
                elif prev_op == "(":
                    # close parenthetical group
                    return result, index
                else:
                    # parenthetical group ending, but we need to close sub-operations within group
                    return result, index - 1
            elif token_text == "(":
                # gather parenthetical group
                right, index = build_eval_tree(
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
                    ) and token_text not in ["**", "^"]:
                        # previous operator is higher priority, so end previous binary op
                        return result, index - 1
                    # get right side of binary op
                    right, index = build_eval_tree(
                        tokens, op_priority, index + 1, depth + 1, token_text
                    )
                    if token_text in ["±", "+/-"]:
                        # See if we need to scale the nominal_value and std_dev terms by an eponent
                        exp_number, exp_number_end = peek_exp_number(tokens, index)
                        if exp_number:
                            result, index = finish_exp_number(
                                tokens,
                                exp_number,
                                exp_number_end,
                                current_token,
                                result,
                                right,
                            )
                            # We know we are not at an ENDMARKER here
                            continue
                    result = EvalTreeNode(
                        left=result, operator=current_token, right=right
                    )
                else:
                    # unary operator
                    right, index = build_eval_tree(
                        tokens, op_priority, index + 1, depth + 1, "unary"
                    )
                    result = EvalTreeNode(left=right, operator=current_token)
        elif token_type == tokenlib.NUMBER or token_type == tokenlib.NAME:
            if result:
                # tokens with an implicit operation i.e. "1 kg"
                if op_priority[""] <= op_priority.get(prev_op, -1):
                    # previous operator is higher priority than implicit, so end
                    # previous binary op
                    return result, index - 1
                right, index = build_eval_tree(
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
                return result, index
            else:
                # recursion all closed, so just return the final result
                return result

        if index + 1 >= len(tokens):
            # should hit ENDMARKER before this ever happens
            raise DefinitionSyntaxError("unexpected end to tokens")

        index += 1

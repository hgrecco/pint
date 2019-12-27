"""
    pint.pint_eval
    ~~~~~~~~~~~~~~

    An expression evaluator to be used as a safe replacement for builtin eval.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

import operator
import token as tokenlib

from .errors import DefinitionSyntaxError

# For controlling order of operations
_OP_PRIORITY = {
    "**": 3,
    "^": 3,
    "unary": 2,
    "*": 1,
    "": 1,  # operator for implicit ops
    "/": 1,
    "+": 0,
    "-": 0,
}

_BINARY_OPERATOR_MAP = {
    "**": operator.pow,
    "*": operator.mul,
    "": operator.mul,  # operator for implicit ops
    "/": operator.truediv,
    "+": operator.add,
    "-": operator.sub,
}

_UNARY_OPERATOR_MAP = {"+": lambda x: x, "-": lambda x: x * -1}


class EvalTreeNode:
    def __init__(self, left, operator=None, right=None):
    """left + operator + right --> binary op
        left + operator --> unary op
        left + right --> implicit op
        left --> single value

    Parameters
    ----------

    Returns
    -------

    """
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

    def evaluate(
        self, define_op, bin_op=_BINARY_OPERATOR_MAP, un_op=_UNARY_OPERATOR_MAP
    ):
        """define_op is a callable that translates tokens into objects
        bin_op and un_op provide functions for performing binary and unary operations

        Parameters
        ----------
        define_op :
            
        bin_op :
             (Default value = _BINARY_OPERATOR_MAP)
        un_op :
             (Default value = _UNARY_OPERATOR_MAP)

        Returns
        -------

        """

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


def build_eval_tree(tokens, op_priority=_OP_PRIORITY, index=0, depth=0, prev_op=None):
    """

    Parameters
    ----------
    Index :
        depth
    Tokens :
        is an iterable of tokens from an expression to be evaluated
    Transform :
        the tokens from an expression into a recursive parse tree
    of :
        operations
    unary :
        ops
    General :
        Strategy
    1 :
        Get left side of operator
    2 :
        If no tokens left
    3 :
        Get operator
    4 :
        Use recursion to create tree starting at token on right side of operator
    4 :
        
    5 :
        Combine left side
    6 :
        Go back to step
    tokens :
        
    op_priority :
         (Default value = _OP_PRIORITY)
    index :
         (Default value = 0)
    depth :
         (Default value = 0)
    prev_op :
         (Default value = None)

    Returns
    -------

    """

    if depth == 0 and prev_op is None:
        # ensure tokens is list so we can access by index
        tokens = list(tokens)

    result = None

    while True:
        current_token = tokens[index]
        token_type = current_token[0]
        token_text = current_token[1]

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

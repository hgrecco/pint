from __future__ import annotations

import pytest

from pint.pint_eval import build_eval_tree, plain_tokenizer, uncertainty_tokenizer
from pint.util import string_preprocessor

TOKENIZERS = (plain_tokenizer, uncertainty_tokenizer)


def _pre(tokenizer, input_text: str, preprocess: bool = False) -> str:
    if preprocess:
        input_text = string_preprocessor(input_text)
    return build_eval_tree(tokenizer(input_text)).to_string()


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
@pytest.mark.parametrize(
    ("input_text", "parsed"),
    (
        ("3", "3"),
        ("1 + 2", "(1 + 2)"),
        ("1 - 2", "(1 - 2)"),
        ("2 * 3 + 4", "((2 * 3) + 4)"),  # order of operations
        ("2 * (3 + 4)", "(2 * (3 + 4))"),  # parentheses
        (
            "1 + 2 * 3 ** (4 + 3 / 5)",
            "(1 + (2 * (3 ** (4 + (3 / 5)))))",
        ),  # more order of operations
        (
            "1 * ((3 + 4) * 5)",
            "(1 * ((3 + 4) * 5))",
        ),  # nested parentheses at beginning
        ("1 * (5 * (3 + 4))", "(1 * (5 * (3 + 4)))"),  # nested parentheses at end
        (
            "1 * (5 * (3 + 4) / 6)",
            "(1 * ((5 * (3 + 4)) / 6))",
        ),  # nested parentheses in middle
        ("-1", "(- 1)"),  # unary
        ("3 * -1", "(3 * (- 1))"),  # unary
        ("3 * --1", "(3 * (- (- 1)))"),  # double unary
        ("3 * -(2 + 4)", "(3 * (- (2 + 4)))"),  # parenthetical unary
        ("3 * -((2 + 4))", "(3 * (- (2 + 4)))"),  # parenthetical unary
        # implicit op
        ("3 4", "(3 4)"),
        # implicit op, then parentheses
        ("3 (2 + 4)", "(3 (2 + 4))"),
        # parentheses, then implicit
        ("(3 ** 4 ) 5", "((3 ** 4) 5)"),
        # implicit op, then exponentiation
        ("3 4 ** 5", "(3 (4 ** 5))"),
        # implicit op, then addition
        ("3 4 + 5", "((3 4) + 5)"),
        # power followed by implicit
        ("3 ** 4 5", "((3 ** 4) 5)"),
        # implicit with parentheses
        ("3 (4 ** 5)", "(3 (4 ** 5))"),
        # exponent with e
        ("3e-1", "3e-1"),
        # multiple units with exponents
        ("kg ** 1 * s ** 2", "((kg ** 1) * (s ** 2))"),
        # multiple units with neg exponents
        ("kg ** -1 * s ** -2", "((kg ** (- 1)) * (s ** (- 2)))"),
        # multiple units with neg exponents
        ("kg^-1 * s^-2", "((kg ^ (- 1)) * (s ^ (- 2)))"),
        # multiple units with neg exponents, implicit op
        ("kg^-1 s^-2", "((kg ^ (- 1)) (s ^ (- 2)))"),
        # nested power
        ("2 ^ 3 ^ 2", "(2 ^ (3 ^ 2))"),
        # nested power
        ("gram * second / meter ** 2", "((gram * second) / (meter ** 2))"),
        # nested power
        ("gram / meter ** 2 / second", "((gram / (meter ** 2)) / second)"),
        # units should behave like numbers, so we don't need a bunch of extra tests for them
        # implicit op, then addition
        ("3 kg + 5", "((3 kg) + 5)"),
        ("(5 % 2) m", "((5 % 2) m)"),  # mod operator
        ("(5 // 2) m", "((5 // 2) m)"),  # floordiv operator
    ),
)
def test_build_eval_tree(tokenizer, input_text: str, parsed: str):
    assert _pre(tokenizer, input_text) == parsed


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
@pytest.mark.parametrize(
    ("input_text", "parsed"),
    (
        ("3", "3"),
        ("1 + 2", "(1 + 2)"),
        ("1 - 2", "(1 - 2)"),
        ("2 * 3 + 4", "((2 * 3) + 4)"),  # order of operations
        ("2 * (3 + 4)", "(2 * (3 + 4))"),  # parentheses
        (
            "1 + 2 * 3 ** (4 + 3 / 5)",
            "(1 + (2 * (3 ** (4 + (3 / 5)))))",
        ),  # more order of operations
        (
            "1 * ((3 + 4) * 5)",
            "(1 * ((3 + 4) * 5))",
        ),  # nested parentheses at beginning
        ("1 * (5 * (3 + 4))", "(1 * (5 * (3 + 4)))"),  # nested parentheses at end
        (
            "1 * (5 * (3 + 4) / 6)",
            "(1 * ((5 * (3 + 4)) / 6))",
        ),  # nested parentheses in middle
        ("-1", "(- 1)"),  # unary
        ("3 * -1", "(3 * (- 1))"),  # unary
        ("3 * --1", "(3 * (- (- 1)))"),  # double unary
        ("3 * -(2 + 4)", "(3 * (- (2 + 4)))"),  # parenthetical unary
        ("3 * -((2 + 4))", "(3 * (- (2 + 4)))"),  # parenthetical unary
        # implicit op
        ("3 4", "(3 * 4)"),
        # implicit op, then parentheses
        ("3 (2 + 4)", "(3 * (2 + 4))"),
        # parentheses, then implicit
        ("(3 ** 4 ) 5", "((3 ** 4) * 5)"),
        # implicit op, then exponentiation
        ("3 4 ** 5", "(3 * (4 ** 5))"),
        # implicit op, then addition
        ("3 4 + 5", "((3 * 4) + 5)"),
        # power followed by implicit
        ("3 ** 4 5", "((3 ** 4) * 5)"),
        # implicit with parentheses
        ("3 (4 ** 5)", "(3 * (4 ** 5))"),
        # exponent with e
        ("3e-1", "3e-1"),
        # multiple units with exponents
        ("kg ** 1 * s ** 2", "((kg ** 1) * (s ** 2))"),
        # multiple units with neg exponents
        ("kg ** -1 * s ** -2", "((kg ** (- 1)) * (s ** (- 2)))"),
        # multiple units with neg exponents
        ("kg^-1 * s^-2", "((kg ** (- 1)) * (s ** (- 2)))"),
        # multiple units with neg exponents, implicit op
        ("kg^-1 s^-2", "((kg ** (- 1)) * (s ** (- 2)))"),
        # nested power
        ("2 ^ 3 ^ 2", "(2 ** (3 ** 2))"),
        # nested power
        ("gram * second / meter ** 2", "((gram * second) / (meter ** 2))"),
        # nested power
        ("gram / meter ** 2 / second", "((gram / (meter ** 2)) / second)"),
        # units should behave like numbers, so we don't need a bunch of extra tests for them
        # implicit op, then addition
        ("3 kg + 5", "((3 * kg) + 5)"),
        ("(5 % 2) m", "((5 % 2) * m)"),  # mod operator
        ("(5 // 2) m", "((5 // 2) * m)"),  # floordiv operator
    ),
)
def test_preprocessed_eval_tree(tokenizer, input_text: str, parsed: str):
    assert _pre(tokenizer, input_text, True) == parsed


@pytest.mark.parametrize(
    ("input_text", "parsed"),
    (
        ("( 8.0 + / - 4.0 ) e6 m", "((8.0e6 +/- 4.0e6) m)"),
        ("( 8.0 ± 4.0 ) e6 m", "((8.0e6 +/- 4.0e6) m)"),
        ("( 8.0 + / - 4.0 ) e-6 m", "((8.0e-6 +/- 4.0e-6) m)"),
        ("( nan + / - 0 ) e6 m", "((nan +/- 0) m)"),
        ("( nan ± 4.0 ) m", "((nan +/- 4.0) m)"),
        ("8.0 + / - 4.0 m", "((8.0 +/- 4.0) m)"),
        ("8.0 ± 4.0 m", "((8.0 +/- 4.0) m)"),
        ("8.0(4)m", "((8.0 +/- 0.4) m)"),
        ("8.0(.4)m", "((8.0 +/- .4) m)"),
        # ("8.0(-4)m", None), # TODO: this should raise an exception
    ),
)
def test_uncertainty(input_text: str, parsed: str):
    if parsed is None:
        with pytest.raises():
            assert _pre(uncertainty_tokenizer, input_text)
    else:
        assert _pre(uncertainty_tokenizer, input_text) == parsed

from __future__ import annotations

import pytest

from pint.pint_eval import plain_tokenizer, uncertainty_tokenizer

VALUES = [
    "1",
    "1 + 2 + 5",
    "10 m",
    "10 metros + 5 segundos",
    "10 metros * (5 segundos)",
]


def _tok(tok, value):
    return tuple(tok(value))


@pytest.mark.parametrize("tokenizer", (plain_tokenizer, uncertainty_tokenizer))
@pytest.mark.parametrize("value", VALUES)
def test_pint_eval(benchmark, tokenizer, value):
    benchmark(_tok, tokenizer, value)

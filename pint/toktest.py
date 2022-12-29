import pint
from pint import Quantity as Q_

import re
import tokenize
from tokenize import NUMBER, STRING, NAME, OP
import token as tokenlib
from io import BytesIO
from pint.pint_eval import _plain_tokenizer, tokenizer, uncertainty_tokenizer
from pint.pint_eval import tokens_with_lookahead

tokenizer = _plain_tokenizer

input_lines = [
    "( 8.0 + / - 4.0 ) e6 m",
    "( 8.0 ± 4.0 ) e6 m",
    "( 8.0 + / - 4.0 ) e-6 m",
    "( nan + / - 0 ) e6 m",
    "( nan ± 4.0 ) m",
    "8.0 + / - 4.0 m",
    "8.0 ± 4.0 m",
    "8.0(4)m",
    "8.0(.4)m",
    "8.0(-4)m", # error!
    "pint == wonderfulness ^ N + - + / - * ± m J s"
]

for line in input_lines:
    result = []
    g = list(uncertainty_tokenizer(line))  # tokenize the string
    for toknum, tokval, _, _, _ in g:
        result.append((toknum, tokval))

    print("====")
    print(f"input line: {line}")
    print(result)
    print(tokenize.untokenize(result))

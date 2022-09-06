from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from B import ClsB

class ClsA:
    def __init__(self, cc: ClsB):
        ...


## from file B.py, it has ClsB but it uses class A
## (file B) from A import ClsA

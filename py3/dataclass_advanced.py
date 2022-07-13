# https://stackoverflow.com/questions/47955263/what-are-data-classes-and-how-are-they-different-from-common-classes

### Example #1

from inspect import signature, getmembers
from dataclasses import dataclass, astuple

from typing import Dict, List, Any
import numpy as np
import sys

# bad
class Data:
    def __init__(self, X: np.ndarray =None, y: np.array=None,
                 kwargs: Dict =None):
        self.X = X
        self.y = y
        self.kwargs = kwargs
    def __repr__(self):
         return self.val
    def __eq__(self, other):
         return self.val == other.val

## good
@dataclass
class Data:
    X: np.ndarray = None  # The field declaration: X
    y: np.array = None    # The field declaration: y
    kwargs: Dict = None   # The field declaration: kwargs

# Example #15
@dataclass
class CrossValidation:
    inner_cv: int
    outer_cv: int
    eval_final_performance: bool = True
    test_size: float = 0.2
    calculate_metrics_per_fold: bool = True
    calculate_metrics_across_folds: bool = False
    outer_folds = None
    inner_folds = dict()
cv1 = CrossValidation(1,2)
cv2 = CrossValidation(1,2)
cv3 = CrossValidation(3,2,test_size=0.5)
print(cv1)
print(cv2)
print(cv3)


# using __slots__

### Example #22
@dataclass
class LoggingState:
    __slots__ =  ['debug', 'info', 'success', 'warning', 'error', 'critical']
    debug: bool
    info: bool
    success: bool
    warning: bool
    error: bool
    critical: bool

logg = LoggingState(debug=False, info=False, success=False, warning=True, error=True, critical=True )
print(f'{logg = }')

# adding a method
### Example #23
@dataclass
class Data():
    X: np.ndarray = None  # The field declaration: X
    y: np.array = None    # The field declaration: y
    z: int = 0   # The field declaration: kwargs
    p: int = 0

    def quad_args(self):
        self.z = self.X ** 4

    def __post_init__(self):
        self.p = self.X ** self.y

d = Data(3,2)
print(d)
print(d.quad_args())
print(d)


## immutable data class
### Example 25
@dataclass(frozen=True)
class Data2():
    X: np.ndarray = 0  # The field declaration: X
    y: np.array = 0    # The field declaration: y
    z: int = 0   # The field declaration: kwargs

d = Data2()
# d.y = 2 # dataclasses.FrozenInstanceError: cannot assign to field 'y'
print(d)

@dataclass(unsafe_hash=True)
class P2D:
    x: int
    y: int

p1 = P2D(1,2)
p2 = P2D(1,2)
p3 = P2D(3,4)

print(p1 == p2)

print(id(p1))
print(id(p2))

tmp_set = set()
tmp_set.add(p1)
tmp_set.add(p2)
tmp_set.add(p3)
print(f'{tmp_set = }') # tmp_set = {P2D(x=1, y=2), P2D(x=3, y=4)}

@dataclass
class Color:
    r : int = 0
    g : int = 0
    b : int = 0

    def __iter__(self):
        yield from astuple(self)

c = Color(1,2,3)
print(*c)

@dataclass
class SlottedColor:
    __slots__ = ["r", "b", "g"]
    r : int
    g : int
    b : int

import sys
print(f"{sys.getsizeof(Color) = }")
print(f"{sys.getsizeof(SlottedColor) = }")

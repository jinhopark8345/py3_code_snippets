from dataclasses import dataclass, field, InitVar
from typing import Type, TypeVar, List, Any, Mapping


@dataclass
class P2D:
    x: float
    y: float


p1 = P2D(1, 3)
print(f"{p1 = }")
print(f"{p1.x = }")
print(f"{p1.y = }")


@dataclass(init=False)
class Point2D:
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


p2 = Point2D(1, 4)
print(f"{p2 = }")
print(f"{p2.x = }")
print(f"{p2.y = }")


@dataclass
class BBox:
    x1: float  # top left x1
    y1: float  # top left y1
    x2: float  # bot right x1
    y2: float  # bot right y1


bbox = BBox(1, 3, 4, 5)
print(f"{bbox = }")


@dataclass(init=False)
class ArgHolder:
    args: List[Any]
    kwargs: Mapping[Any, Any]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


a = ArgHolder(1, 2, three=3)
print(f"{a = }")


# __post_init__
@dataclass()
class Student:
    name: str
    clss: int
    stu_id: int
    marks: []
    avg_marks: float = field(init=False)

    def __post_init__(self):
        self.avg_marks = sum(self.marks) / len(self.marks)


st = Student("HTD", 10, 17, [98, 85, 90])
print(f"{st = }")


# init-only variables

# @dataclass
# class C:
#     i: int
#     j: int = None
#     database: InitVar[DatabaseType] = None

#     def __post_init__(self, database):
#         if self.j is None and database is not None:
#             self.j = database.lookup('j')

# c = C(10, database=my_database)

# inheritance
@dataclass
class Base:
    x: Any = 15.0
    y: int = 0


@dataclass
class C(Base):
    z: int = 10
    x: int = 15
    mylist: list = field(default_factory=list)


c = C(3, 4)
print(f"{c = }")


@dataclass
class DF:
    a: int = 10
    b: int = 15
    c: list = field(default_factory=list)


df = DF(1, 2, [1, 2, 3])
print(f"{df = }")
df2 = DF(1, 2)
print(f"{df2 = }")

# Mutable default values


class CC:
    x = []

    def add(self, element):
        self.x.append(element)


o1 = CC()
o2 = CC()
o1.add(1)
o2.add(2)
assert o1.x == [1, 2]
assert o1.x is o2.x


@dataclass
class D:
    x: list = field(default_factory=list)


assert D().x is not D().x


# Descriptor-typed fields¶


class IntConversionDescriptor:
    def __init__(self, *, default):
        self._default = default

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, obj, type):
        # print(f'__get__ called: {obj = }')
        print(f'__get__ called')
        if obj is None:
            return self._default

        return getattr(obj, self._name, self._default)

    def __set__(self, obj, value):
        # print(f'__set__ called: {obj, value = }')
        print(f'__set__ called, {value = }')
        setattr(obj, self._name, int(value))


@dataclass
class InventoryItem:
    quantity_on_hand: IntConversionDescriptor = IntConversionDescriptor(
        default=100
    )


i = InventoryItem()
print(i.quantity_on_hand)  # 100
i.quantity_on_hand = 2.5  # calls __set__ with 2.5
print(i.quantity_on_hand)  # 2

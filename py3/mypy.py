
class Person_wrong:
    name: str
    age: int

    def __init__(self, name, age): # no typing hint -> mypy doesn't do anything
        self.name = name
        self.age = age
        reveal_type(self.name)


class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


# p = Person(5, 5)  #  Argument 1 to "Person" has incompatible type "int"; expected "str"
p = Person("asd", 5)

p2= Person_wrong("asd", 5)
# mypy.py:9: note: Revealed type is "Any"
# mypy.py:9: note: 'reveal_type' always outputs 'Any' in unchecked functions

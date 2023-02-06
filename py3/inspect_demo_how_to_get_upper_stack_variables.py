# ref : https://stackoverflow.com/questions/6618795/get-locals-from-calling-namespace-in-python
import inspect
from pprint import pp, pprint


class A:
    def __init__(self):
        self.image_file_name = 'jinho'
        self.something_else = 123
    def __repr__(self):
        repr = ""


        return str(self.__dict__)

def show_callers_locals():
    """Print the local variables in the caller's frame."""
    frame = inspect.currentframe()
    try:
        pprint("------------ one level up ------------ ")
        pprint(frame.f_back.f_locals, indent=4)

        pprint("------------ two level up ------------ ")
        pp(frame.f_back.f_back.f_locals, indent=4)

        pprint("------------ three level up ------------ ")
        pprint(frame.f_back.f_back.f_back.f_locals, indent=4)
    finally:
        del frame


def first():
    a = A()
    second()

def second():
    b = 4
    show_callers_locals()
    print()


first()

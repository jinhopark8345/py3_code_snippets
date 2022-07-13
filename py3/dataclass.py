
from dataclasses import dataclass

@dataclass
class InventoryItem:
    """Class for keeping track of an item in inventory."""
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand

"""
# Will add, among other things, a __init__() that looks like:

# def __init__(self, name: str, unit_price: float, quantity_on_hand: int=0):
#     self.name = name
#     self.unit_price = unit_price
#     self.quantity_on_hand = quantity_on_hand
"""

ii = InventoryItem("table", 1000, 3)
print(f'{ii = }')

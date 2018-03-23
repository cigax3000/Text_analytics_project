import location

class Item(object):

    def __init__(self, name, weight, current_warehouse):
        self.name = name
        self.weight = weight
        self.current_location = current_warehouse
        current_warehouse.add_item_to_storage(self)

item_a = Item('Piano', 1000, location.warehouse_a)
item_b = Item('Sofa', 200, location.warehouse_b)
item_c = Item('Table', 150, location.warehouse_c)
item_d = Item('Jacket', 10, location.warehouse_d)
item_e = Item('CD', 1, location.warehouse_e)
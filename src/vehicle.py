import location

class Vehicle(object):

    def __init__(self, name, current_location):
        self.name = name
        self.current_location = current_location

    def move_to(self, location):
        if location in self.current_location.roads_to:
            self.current_location = location
            print('{} moves to {}.'.format(self.name, location.name))
        else:
            print('{} cannot find road to {}.'.format(self.name, location.name))

class Truck(Vehicle):

    def __init__(self, name, current_location, max_capacity):
        super().__init__(name, current_location)
        self.max_capacity = max_capacity
        self.storage_space = list()

    def get_storage_weight(self):
        return sum([item.weight for item in self.storage_space])

    def get_remaining_capacity(self):
        return self.max_capacity - self.get_storage_weight()

    def add_item_to_storage(self, item):
        if item.current_location == self.current_location:
            if self.get_remaining_capacity() > item.weight:
                self.storage_space.append(item)
                print('{} loads {}'.format(self.name, item.name))
                item.current_location = self
            else:
                print('{} is too heavy for remaining capacity of {}.'.format(item.name, self.name))
        else:
            print('{} cannot find {} at {}.'.format(self.name, item.name, self.current_location.name))

    def remove_item_from_storage(self, item):
        if item in self.storage_space:
            self.storage_space.remove(item)

truck_a = Truck('Truck A', location.warehouse_a, 100)
truck_b = Truck('Truck B', location.warehouse_b, 500)
truck_c = Truck('Truck C', location.warehouse_c, 1000)

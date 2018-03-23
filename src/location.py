class Location(object):

    def __init__(self, name):
        self.name = name
        self.roads_to = list()

    def add_road_to(self, location):
        self.roads_to.append(location)
        location.roads_to.append(self)


class Warehouse(Location):

    def __init__(self, name):
        super().__init__(name)
        self.storage_space = list()

    def add_item_to_storage(self, item):
        self.storage_space.append(item)

    def remove_item_from_storage(self, item):
        if item in self.storage_space:
            self.storage_space.remove(item)


warehouse_a = Warehouse('Warehouse A')
warehouse_b = Warehouse('Warehouse B')
warehouse_c = Warehouse('Warehouse C')
warehouse_d = Warehouse('Warehouse D')
warehouse_e = Warehouse('Warehouse E')

warehouse_a.add_road_to(warehouse_b)
warehouse_a.add_road_to(warehouse_c)
warehouse_b.add_road_to(warehouse_c)
warehouse_d.add_road_to(warehouse_c)
warehouse_e.add_road_to(warehouse_c)
warehouse_d.add_road_to(warehouse_e)
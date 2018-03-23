from item import item_a, item_b, item_c, item_d, item_e
from location import warehouse_a, warehouse_b, warehouse_c, warehouse_d, warehouse_e
from vehicle import truck_a, truck_b, truck_c

def main():
    # This should not work.
    truck_a.add_item_to_storage(item_a)

    # This should work.
    truck_a.move_to(warehouse_c)

if __name__ == '__main__':
    main()
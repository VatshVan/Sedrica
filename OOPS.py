class Car:
    def __init__(self, make, model, price):
        self.make = make
        self.model = model
        self.price = price

    def apply_discount(self, discount):
        """Apply a discount percentage to the car price."""
        self.price -= self.price * (discount / 100)

    def get_info(self):
        """Display car information."""
        print(f"Car: {self.make} {self.model}, Price: ${self.price:.2f}")


car1 = Car("Toyota", "Corolla", 25000)
car1.get_info()
car1.apply_discount(10)
car1.get_info()

class Car:
    def __init__(self, make, model, price):
        self.make = make
        self.model = model
        self.__price = price

    def apply_discount(self, discount):
        self.__price -= self.__price * (discount / 100)

    def get_info(self):
        print(f"Car: {self.make} {self.model}, Price: ${self.__price:.2f}")

    def get_price(self):
        return self.__price

    def set_price(self, price):
        if price > 0:
            self.__price = price
        else:
            print("Price must be positive.")


car2 = Car("Ford", "Mustang", 40000)
car2.get_info()
car2.set_price(38000)
print("Updated Price:", car2.get_price())

class ElectricCar(Car):
    def __init__(self, make, model, price, battery_range):
        super().__init__(make, model, price)
        self.battery_range = battery_range

    def get_info(self):
        print(f"Electric Car: {self.make} {self.model}, Price: ${self.get_price():.2f}, "
              f"Battery Range: {self.battery_range} miles")

class SportsCar(Car):
    def __init__(self, make, model, price, top_speed):
        super().__init__(make, model, price)
        self.top_speed = top_speed

    def get_info(self):
        print(f"Sports Car: {self.make} {self.model}, Price: ${self.get_price():.2f}, "
              f"Top Speed: {self.top_speed} mph")

inventory = [
    Car("Toyota", "Camry", 30000),
    ElectricCar("Tesla", "Model 3", 50000, 350),
    SportsCar("Ferrari", "488 Spider", 300000, 211)
]

for car in inventory:
    car.get_info()

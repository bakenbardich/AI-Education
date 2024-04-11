class Client:
    default_name = "Andrew"
    default_age = 30

    def __init__(self, name=None, age=None, bank=None):
        self.name = name or self.default_name
        self.age = age or self.default_age
        self.bank = bank
        self.__account = None
        self.__house = None
        if bank:
            self.__account = self.add_account()

    def add_account(self):
        if not self.bank:
            return None
        account = self.bank.add_account(self)
        return account

    def info(self):
        if self.__account:
            print(
                f"Name: {self.name}, Age: {self.age}, House: {self.__house}, Account Balance: {self.__account.amount}")
        else:
            print("No account linked.")

    def __str__(self):
        return f"Name: {self.name}, Age: {self.age}, House: {self.__house}, Account Balance: {self.__account.amount if self.__account else 'No account linked'}"

    @staticmethod
    def default_info():
        print(f"Default Name: {Client.default_name}, Default Age: {Client.default_age}")

    def make_deal(self, house, price):
        if self.__account.amount >= price:
            self.__account -= price
            self.__house = house
        else:
            print("Not enough funds for the deal!")

    def earn_money(self, amount):
        self.__account += amount

    def buy_house(self, house, discount=0):
        price = house.final_price(discount)
        if self.__account.amount >= price:
            self.make_deal(house, price)
        else:
            print("Not enough funds for the house!")


class Account:
    default_amount = 0

    def __init__(self, client, amount=default_amount):
        self.client = client
        self.amount = amount

    def __str__(self):
        return f"Account owner: {self.client.name}, Balance: {self.amount}"

    def __iadd__(self, other):
        self.amount += other
        return self

    def __isub__(self, other):
        if self.amount - other >= 0:
            self.amount -= other
        else:
            print("Insufficient funds!")
        return self


class Bank:
    def __init__(self):
        self.__accounts = []

    def add_account(self, client: Client) -> Account:
        for account in self.__accounts:
            if account.client == client:
                raise ValueError("Client already has an account!")
        new_account = Account(client)
        self.__accounts.append(new_account)
        return new_account

    def __str__(self):
        bank_info = "Bank Clients:\n"
        for account in self.__accounts:
            bank_info += str(account) + "\n"
        return bank_info


class House:
    def __init__(self, area, price):
        self._area = area
        self._price = price

    def final_price(self, discount):
        return self._price - discount


class SmallHouse(House):
    def __init__(self, price):
        super().__init__(area=42, price=price)

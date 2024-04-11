from task_oop import Client
from task_oop import Bank
from task_oop import SmallHouse

# 1
green_bank = Bank()
red_bank = Bank()
# 2
Client.default_info()
# 3
student = Client(bank=green_bank)
# 4
teacher = Client(bank=red_bank)
# 5
student.info()
# 6
print(teacher)
# 7
sh = SmallHouse(price=1000)
# 8
student.buy_house(sh)
# 9
student.earn_money(10000)
# 10
student.buy_house(sh)
# 11
student.info()

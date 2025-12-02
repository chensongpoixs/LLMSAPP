"""
author: chensong
date: 2025/12/02 19:00
file: forwhile.py
desc: for 和 while 循环

"""

# for 打印列表

my_list = [10, 20, 30, 40, 50];

for item in my_list:
    print(item);

# range 0~9

#for i in range(my_list):
  #  print(i);

# range 0~4
for i in range(5):
    print(i);


# while 打印列表
index = 0;

while index < len(my_list):
    print(my_list[index]);
    index += 1;
# while 计算 1~100 之间的和
sum = 0;
num = 1;

while num <= 100:
    sum += num;
    num += 1;
print("1~100 之间的和为:", sum);



#for 的步长为2
for i in range(0, 10, 2):
    print(i);



def func():
    print("hello world");

def func2(a, b):
    return a + b;

def list_add(a, b):
    return a , func2(a, b);




func();



# 列表, 长度
list, len =  list_add([1, 2, 3], [9, 1, 3]);

print(list, len);



# python  中类 的定义
class Person:
    name = "";
    age = 0;
    # 构造函数
    def __init__(self, name, age):
        self.name = name;
        self.age = age;

    def display(self):
        print("Name:", self.name, " Age:", self.age);   
# 创建对象
p1 = Person("Alice", 30);
p1.display();
p2 = Person("Bob", 25);
p2.display();


# pyton 中继承
class Student(Person):
    student_id = "";
    def __init__(self, name, age, student_id):
        super().__init__(name, age);
        self.student_id = student_id;

    def display(self):
        super().display();
        print("Student ID:", self.student_id);

s1 = Student("Charlie", 20, "S12345");
s1.display();

# 异常处理
try:
    result = 10 / 0;    
except ZeroDivisionError as e:
    print("Error: Division by zero is not allowed.");
finally:
    print("Execution completed.");  

# 6、 可变变量和不可变量的区别

list1 = [1, 2, 3];

list2 = list1;

list2[0] = 10;

print("list1:", list1);
print("list2:", list2);


# python 分片：a[::]
list3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

print('list3[::2]= ', list3[::2]);  # 输出: [0, 2, 4, 6, 8]
print('list3[1::2] = ',list3[1::2]);  # 输出: [1, 3, 5, 7, 9]
print('list3[::-1] = ' , list3[::-1]);  # 输出: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
#完整的切片语法是 a[start:stop:step]，其中 start 是起始索引，stop 是结束索引（不包含），step 是步长。


# 列表推导式
original_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
squared_list = [x**2 for x in original_list if x % 2 == 0];
print("Squared even numbers:", squared_list);
# 输出: Squared even numbers: [4, 16, 36, 64, 100]  


list4 = [x for x in range(20) if x % 3 == 0];
print("Multiples of 3:", list4);
# 输出: Multiples of 3: [0, 3, 6, 9, 12, 15, 18]
# 字典推导式
original_dict = {'a': 1, 'b': 2, 'c': 3};
squared_dict = {k: v**2 for k, v in original_dict.items()};
print("Squared dictionary:", squared_dict);
# 输出: Squared dictionary: {'a': 1, 'b': 4, 'c': 9}
# 集合推导式
original_set = {1, 2, 3, 4, 5};
squared_set = {x**2 for x in original_set if x % 2 != 0};
print("Squared odd numbers set:", squared_set);
# 输出: Squared odd numbers set: {1, 9, 25}
# 元组推导式（生成器表达式）
original_tuple = (1, 2, 3, 4, 5);
squared_tuple = tuple(x**2 for x in original_tuple if x > 2);
print("Squared numbers greater than 2 tuple:", squared_tuple);
# 输出: Squared numbers greater than 2 tuple: (9, 16, 25)
# 使用列表推导式生成一个包含1到10之间所有偶数的列表
#even_numbers = [x for x in range(1, 11) if x %== 0];
# print("Even numbers from 1 to 10:", even_numbers)
# 
# 
# 使用with语句打开一个文件，并使用列表推导式读取文件中的所有行，去除行末的换行符，生成一个新的列表
# with open('example.txt', 'r') as file:
#     lines = [line.strip() for line in file]
#   print("Lines from file:", lines)
#   # 使用字典推导式创建一个字典，键为1到5的数字，值为这些数字的平方
#   squared_dict = {x: x**2 for x in range(1, 6)}
#   print("Squared dictionary:", squared_dict)
#   # 使用集合推导式创建一个集合，包含1到20之间所有能被3整除的数字       


with open('test_with.log', 'w') as file:
    file.write("This is a test log file.\n")
    file.write("It contains multiple lines of text.\n")
    file.write("End of the log file.\n")

# 函数解包
def unpack_example(a, b, c):
    return a + b + c;

result = unpack_example(*[1, 2, 3]);
print("Unpack result:", result);    
# 输出: Unpack result: 6
result_dict = unpack_example(**{'a': 4, 'b': 5, 'c': 6});
print("Unpack dict result:", result_dict);  
# 输出: Unpack dict result: 15


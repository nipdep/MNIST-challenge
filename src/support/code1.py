#define global variable
num1 = 1

# use global variable
def func0():
    val = 3*num1
    return val

#define a global variable in a functions local scope
def func1():
    global num2
    num2 = 4

# use global variable define by func1
def func2():
    val = 3*num2
    return val

print(f'func0 result: {func0()}')
## uncomment this line ,but I'll give the same error as you get
#print(f'func2 result before run func1: {func2()}')
func1()
print(f'func2 result after run func1: {func2()}')

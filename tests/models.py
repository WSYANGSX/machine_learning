x = "123456"
print(x, end="")
print(f"\033[{len(x)}D", end="")
print("\t"*2, "1234567")

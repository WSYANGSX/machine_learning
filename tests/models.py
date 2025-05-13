class A:
    parser = {}

    def __init__(self):
        pass

    @property
    def keys(self) -> list:
        return list(self.parser.keys())

    def register(self, name, cl):
        self.parser.update({name: cl})


class B:
    def __init__(self):
        pass

    def info(self):
        print("我是B")


class C:
    def __init__(self):
        pass

    def info(self):
        print("我是C")


a = A()

a.register(name="b", cl=B)
a.register(name="c", cl=C)

print(a.keys)
b = a.parser["b"]()
b.info()
c = a.parser["c"]()
c.info()

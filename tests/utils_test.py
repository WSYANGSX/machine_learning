class Parent:
    class_attr = "父类的类属性"


class Child(Parent):
    def instance_method(self):
        print(self.__class__.class_attr)  # 继承父类的类属性


child = Child()
child.instance_method()  # 输出: 父类的类属性


class Child(Parent):
    class_attr = "子类的类属性"  # 覆盖父类的类属性

    def instance_method(self):
        print(Parent.class_attr)  # 输出子类的类属性


child = Child()
child.instance_method()  # 输出: 子类的类属性

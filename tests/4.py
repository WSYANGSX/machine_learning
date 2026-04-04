params = {"size0": [1, 2], "gg": 4.5, "img": 6, "pig": 5}
print(id(params))


def fun1(size0, **params):
    size0.append(50)
    print(size0)


def fun2(gg, **params):
    params["img"] = 100
    print(id(params))
    params.update({"4": 5})
    print(params)


def fun3(img, **params):
    print(id(img))


def fun4(size0, **params):
    print(size0)


def fun5(params):
    params.update({"4": 5})


def main():
    fun1(**params)
    fun2(**params)
    fun3(**params)
    fun4(**params)
    fun5(params)

    print(params)


if __name__ == "__main__":
    main()

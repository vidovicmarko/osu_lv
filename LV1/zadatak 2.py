try:
    number = float(input("Unesi ocjenu: "))
    if number < 0 or number > 1.0:
        raise ValueError
    elif number < 0.6:
        print("F")
    elif number < 0.7:
        print("D")
    elif number < 0.8:
        print("C")
    elif number < 0.9:
        print("B")
    else:
        print("A")
except ValueError:
    print("Mora biti unešen broj između 0 i 1.0!")

numbers = []

while (True):
    try:
        number = input("Unesi broj: ")
        if number == "Done":
            break
        numbers.append(float(number))
    except:
        print("Potrebno je unjeti broj tipa float!")

print(f"Količina brojeva: {len(numbers)}")
print(f"Srednja vrijednost: {sum(numbers) / len(numbers)}")
print(f"Minimalna vrijednost: {min(numbers)}")
print(f"Maksimalna vrijednost: {max(numbers)}")

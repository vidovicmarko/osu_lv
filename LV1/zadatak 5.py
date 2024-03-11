ham = []
spam = []

file = open("SMSSpamCollection.txt")
for row in file:
    row = row.rstrip()
    type = row.split("\t")
    if type[0] == "ham":
        ham.append(type[1])
    else:
        spam.append(type[1])


def average_word_count(list):
    total = 0
    for word in list:
        total += len(word.split())
    return round(total / len(list),2)

def total_ends_with(list):
    total = 0
    for word in list:
        if word.endswith("!"):
            total += 1
    return total


print(f"Prosječan broj riječi u porukama tipa ham: {average_word_count(ham)}")
print(f"Prosječan broj riječi u porukama tipa spam: {average_word_count(spam)}")
print(f"Broj sms poruka tipa spam koje završavaju sa uskličnikom: {total_ends_with(spam)}")

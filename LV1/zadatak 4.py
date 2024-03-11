words_total = {}

file = open("song.txt")
for row in file:
    row = row.rstrip()
    words = row.split(" ")
    for word in words:
        if word not in words_total:
            words_total[word] = 1
            continue
        words_total[word] = words_total[word] + 1
file.close()

once = 0

for word in words_total:
    if words_total[word] == 1:
        once += 1
        print(f"{word} : {words_total[word]}")
print(once)

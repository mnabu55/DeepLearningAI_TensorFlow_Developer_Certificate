import csv

with open("sample.csv", "r") as f:
    #print(f.read())

    reader = csv.reader(f, delimiter=',')
    next(reader)
    for row in reader:
        print(row)
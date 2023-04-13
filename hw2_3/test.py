import csv

if __name__ == "__main__":

    cnt = 0
    acc_cnt = 0
    file = open("./pred.csv")
    reader = csv.reader(file)
    header = next(reader)
    file1 = []
    file2 = []
    for a,b in reader:
       file1.append((a,b))
    file.close()

    file = open("./hw2_data/hw2_data/digits/usps/val.csv")
    reader = csv.reader(file)
    header = next(reader)
    for a, b in reader:
        file2.append((a, b))
    file.close()
    total=0
    cnt = 0
    for i in range(len(file1)):
        for j in range(len(file2)):
            if file1[i][0] == file2[j][0]:
                total += 1
                if file1[i][1] == file2[j][1]:
                    cnt +=1
                break
    print('acc:', cnt/total,"correct",cnt,"total",total)

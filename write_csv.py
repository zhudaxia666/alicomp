import csv

data=[[0,0,0,0,1],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,1]]
csvfile=open('csvfile.csv','w',newline='')
write1=csv.writer(csvfile)
for i in data:
    write1.writerow(i)
csvfile.close()
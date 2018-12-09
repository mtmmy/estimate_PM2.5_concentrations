import csv
from numpy import linalg
import numpy as np
import random
import matplotlib.pyplot as plt


David = []
Davidtest = []
with open('training.csv') as csv_file:
	csv_reader = csv.reader(csv_file)
	for row in csv_reader:
		David.append([(float(row[0])+float(row[1])+float(row[2]))/3,float(row[3]),1])
with open('testing.csv') as csv_file:
	csv_reader = csv.reader(csv_file)
	for row in csv_reader:
		Davidtest.append([(float(row[0])+float(row[1])+float(row[2]))/3,float(row[3]),0])
		David.append([(float(row[0])+float(row[1])+float(row[2]))/3,float(row[3]),0])
David.sort(key = lambda x:x[0])


num = []
val = []
for k in [1,3,5,10,50,100]:
	num.append(k)
	RMSE = 0
	for i in range(len(Davidtest)):
		index = David.index(Davidtest[i])
		count,ans,j = 0,0,0
		if index <= int(k/2):
			while count < k:
				if David[j][2] == 1:
					ans += David[j][1]
					count += 1
				j += 1
		elif index >= len(David)-int(k/2):
			while count < k:
				if David[len(David)-1-j][2] == 1:
					ans += David[len(David)-1-j][1]
					count += 1
				j += 1
		else:
			while count < k:
				if index-int(k/2)+j < len(David)-1 and David[index-int(k/2)+j][2] == 1:
					ans += David[index-int(k/2)+j][1]
					count += 1
				j += 1
		ans /= count
		RMSE += (Davidtest[i][1]-ans)**2
	RMSE = (RMSE/len(Davidtest))**(1/2)
	val.append(RMSE)

	print("k = ", k, " RMSE = ", RMSE)

print(num)
print(val)
num = np.array(num)
val = np.array(val)
plt.plot(num,val)
plt.plot(num,val,'*')
plt.show()
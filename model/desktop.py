import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df1 = []
df = pd.read_csv('/Users/ken/Desktop/data.csv')
df1 = df
#print(df1)

df['max'] = df[(df.shift(1) <= df) & (df.shift(-1) <= df) & (df > 2)]
dfm = df['max']
dfy = df['Y']
for i in range(len(dfm)) :
    if not dfm[i] > 2 :
        dfm[i] = 0

plt.scatter(df.index, df['max'], c='g')
df.plot()
listmax = list(df[(df.shift(1) < df) & (df.shift(-1) < df) & (df > 2)].index.values)

dfindex = []
for i in range(len(dfm)):
    if dfm[i] > 2 :
        dfindex.append(i)
    
max = 0
for i in range(len(dfindex) - 1) :
    a = dfindex[i + 1] - dfindex[i]
    if a > max:
        max = a
for i in range(len(dfindex) - 1):
    b = dfindex[i + 1] - dfindex[i]




dfrealindex = []
for i in range(len(dfindex)-1):
    if dfindex[i + 1] - dfindex[i] > 50:
        dfrealindex.append(dfindex[i])

dfstep = [[0] * 410 for _ in range(413)]
dfchoose = []
k = 0
l = 0
for i in range(len(dfy)):
    for j in range(413):
        if i < dfrealindex[j + 1] and i >= dfrealindex[j]:
            dfchoose.append(j)
for i in range(len(dfchoose) ):
    if dfchoose[i] > k:
        l = 0
        k += 1
        dfstep[k][l] = dfy[i+3424]
    else:
        dfstep[k][l] = dfy[i+3424]
        l += 1    



data_number = []
for i in range(413):
    dnumber = dfrealindex[i + 1] - dfrealindex[i]
    data_number.append(dnumber)
data_number_avg = []
for i in range(413):
    data_number_avg.append((data_number[i])/100)

final = [[0] * 100 for _ in range(414)]
for i in range(len(dfrealindex) - 1):
    dfstep_1 = dfstep[i]
    myinter = []
    for j in range(0, 100):
        interpolate = j * data_number_avg[i]
        interpolate1 = int(interpolate)
        interpolate2 = interpolate1 + 1
        slope = dfy[interpolate2 + dfrealindex[i]] - dfy[interpolate1 + dfrealindex[i]]
        myinterpolate = dfy[interpolate1 + dfrealindex[i]] + slope * (interpolate - interpolate1)
        myinter.append(myinterpolate)
        final[i][j] = myinterpolate
    AAA = np.linspace(0,99,100)
    plt.plot(AAA, myinter)
plt.show()
#rint(final)



#import csv
#final_row = np.transpose(final)
#title = []
#with open('output.csv','w',newline = '') as csvFile:
    #writer = csv.writer(csvFile,delimiter = ';')
    #for i in range(414):
        #writer.writerow(final[i])


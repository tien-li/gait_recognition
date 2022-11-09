import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel('/Users/ken/Desktop/專題/20200428/experiment_data0428/林天立_xlsx/Normal.xlsx',sheet_name='right')


df['max'] = df[(df.shift(1) <= df) & (df.shift(-1) <= df) & (df > 2)]
dfm = df['max']
dfy = df['Y']
for i in range(len(dfm)) :
    if not dfm[i] > 2 :
        dfm[i] = 0
#plt.scatter(df.index, dfm, c='g')
#plt.show()


dfindex = [] #存取極大值的index
for i in range(len(dfm)):
    if dfm[i] > 2 :
        dfindex.append(i)


max = 0
for i in range(len(dfindex) - 1) : #找出數據最多
    a = dfindex[i + 1] - dfindex[i]
    if a > max:
        max = a


dfrealindex = [] #將極大值濾波
for i in range(len(dfindex)-1):
    if dfindex[i + 1] - dfindex[i] > 50:
        dfrealindex.append(dfindex[i])


step_number = len(dfrealindex) - 1
dfstep = [[0] * max for _ in range(step_number)] #每一步存進一列
dfchoose = [] #每一數據屬於第幾步
k = 0
l = 0
for i in range(len(dfy)):
    for j in range(step_number):
        if i < dfrealindex[j + 1] and i >= dfrealindex[j]:
            dfchoose.append(j)


for i in range(len(dfchoose) ):
    if dfchoose[i] > k:
        l = 0
        k += 1
        dfstep[k][l] = dfy[i+dfrealindex[0]]
    else:
        dfstep[k][l] = dfy[i+dfrealindex[0]]
        l += 1    


data_number = [] #存取兩步數之間的差
for i in range(step_number):
    dnumber = dfrealindex[i + 1] - dfrealindex[i]
    data_number.append(dnumber)


data_number_avg = [] #分成100等分
for i in range(step_number):
    data_number_avg.append((data_number[i])/100)


final = [[0] * 100 for _ in range(step_number)] #內差後的陣列
for i in range(step_number):
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
    #plt.plot(AAA, myinter)
#plt.show()

from pandas import DataFrame
F = DataFrame(final)
F.to_excel('/Users/ken/Desktop/專題/20200428/experiment_data0428/林天立_output/Normal_right.xlsx')
print('123')
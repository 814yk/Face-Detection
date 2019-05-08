import numpy as np
import cv2
import os
import math
import random


filepath = str(os.getcwd()).replace("\\","/")+"/venv/FDDB-folds"
imagepath = str(os.getcwd()).replace("\\","/")+"/venv/originalPics/"
skinpath = str(os.getcwd()).replace("\\","/")+"/venv/skin/"
nonskinpath = str(os.getcwd()).replace("\\","/")+"/venv/nonskin/"
outputpath = str(os.getcwd()).replace("\\","/")+"/venv/"
resultpath=str(os.getcwd()).replace("\\","/")+"/venv/test/bayesian/RGB/"

gtFileName =[]
tpList=[]
fileList = []
skinList = []


skinAvgListR=[]
nonskinAvgListR=[]
skinAvgListG=[]
nonskinAvgListG=[]
skinAvgListB=[]
nonskinAvgListB=[]

cskinAvgListR=[]
cnonskinAvgListR=[]
cskinAvgListG=[]
cnonskinAvgListG=[]
cskinAvgListB=[]
cnonskinAvgListB=[]


rgbList=[]

skinAvgR=0
skinVarR=0
skinAvgG=0
skinVarG=0
skinAvgB=0
skinVarB=0

nonskinAvgR=0
nonskinVarR=0
nonskinAvgG=0
nonskinVarG=0
nonskinAvgB=0
nonskinVarB=0

cnt = 1
prior = 0
'''
for i in range(0,9):
    gtFileName.append("FDDB-fold-0"+str(i+1))
gtFileName.append("FDDB-fold-10")

for i in range(0,10):
    f = open(filepath + '/' + gtFileName[i] + '.txt', 'r', encoding="utf-8")
    lines = f.read().split()
    fileList.append(lines)
    f.close()

for i in range(0,10):
    for j in range(0, len(fileList[i])):
        skinList.append(nonskinpath+fileList[i][j])
#print(skinList)

for imgName in skinList:
    tempSumR = 0
    tempSumG = 0
    tempSumB = 0
    count = 0
    img = cv2.imread(imgName+".jpg", cv2.IMREAD_COLOR)
    rows, cols, channel = img.shape
    blueImg = img[:, :, 0]
    greenImg = img[:, :, 1]
    redImg = img[:, :, 2]
    for row in range(rows):
        for col in range(cols):
            k = redImg[row, col]
            l = blueImg[row, col]
            m = greenImg[row, col]
            if k != 0 and l !=0 and m !=0:
                tempSumR += k
                tempSumG += m
                tempSumB += l
                count += 1
    skinAvgListR.append(int(tempSumR/count))
    skinAvgListG.append(int(tempSumG / count))
    skinAvgListB.append(int(tempSumB / count))
    f1 = open(outputpath+"nonkinAvgR.txt",'a')
    f2 = open(outputpath + "nonskinAvgG.txt", 'a')
    f3 = open(outputpath + "nonskinAvgB.txt", 'a')
    f1.write(imgName+".jpg"+','+str(int(tempSumR/count))+'\n')
    f2.write(imgName + ".jpg" + ',' + str(int(tempSumG / count)) + '\n')
    f3.write(imgName + ".jpg" + ',' + str(int(tempSumB / count)) + '\n')
    f1.close()
    f2.close()
    f3.close()
    print("loop : ",cnt)
    cnt+=1
'''

#print(skinAvgList)
#############################################


f = open(outputpath+'skinAvgR.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    skinAvgListR.append([skinpath + lines[i].split(',')[0] + '.jpg', lines[i].split(',')[1]])
print(skinAvgListR)

f = open(outputpath+'nonskinAvgR.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    nonskinAvgListR.append([nonskinpath + lines[i].split(',')[0] + '.jpg', lines[i].split(',')[1]])
print(nonskinAvgListR)


f = open(outputpath+'skinAvgG.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    skinAvgListG.append([skinpath + lines[i].split(',')[0] + '.jpg', lines[i].split(',')[1]])
print(skinAvgListG)

f = open(outputpath+'nonskinAvgG.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    nonskinAvgListG.append([nonskinpath + lines[i].split(',')[0] + '.jpg', lines[i].split(',')[1]])
print(nonskinAvgListG)


f = open(outputpath+'skinAvgB.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    skinAvgListB.append([skinpath + lines[i].split(',')[0] + '.jpg', lines[i].split(',')[1]])
print(skinAvgListB)

f = open(outputpath+'nonskinAvgB.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    nonskinAvgListB.append([nonskinpath + lines[i].split(',')[0] + '.jpg', lines[i].split(',')[1]])
print(nonskinAvgListB)

######################################
randIndex=[]
ran_num = random.randint(0,len(skinAvgListB)-1)
for i in range(0,len(skinAvgListB)):
    while ran_num in randIndex:
        ran_num = random.randint(0,len(skinAvgListB)-1)
    randIndex.append(ran_num)

trainingSkinListR = []
trainingSkinListG = []
trainingSkinListB = []
trainingNonSkinListR = []
trainingNonSkinListG = []
trainingNonSkinListB = []
for i in range(0,1993):
    idx = randIndex[i]
    trainingSkinListR.append(skinAvgListR[idx])
    trainingSkinListG.append(skinAvgListG[idx])
    trainingSkinListB.append(skinAvgListB[idx])
    trainingNonSkinListR.append(nonskinAvgListR[randIndex[i]])
    trainingNonSkinListG.append(nonskinAvgListG[randIndex[i]])
    trainingNonSkinListB.append(nonskinAvgListB[randIndex[i]])
testSkinListR = []
testSkinListG = []
testSkinListB = []
testNonSkinListR = []
testNonSkinListG = []
testNonSkinListB = []

for i in range(1993, 2845):
    idx = randIndex[i]
    testSkinListR.append(skinAvgListR[idx])
    testSkinListG.append(skinAvgListG[idx])
    testSkinListB.append(skinAvgListB[idx])
    testNonSkinListR.append(nonskinAvgListR[randIndex[i]])
    testNonSkinListG.append(nonskinAvgListG[randIndex[i]])
    testNonSkinListB.append(nonskinAvgListB[randIndex[i]])
####################################

for i in trainingSkinListB:
    skinAvgB+=int(i[1])
skinAvgB = int(skinAvgB/len(trainingSkinListB))

for i in trainingSkinListB:
    skinVarB += (int(i[1])-skinAvgB)**2
skinVarB = int(skinVarB/len(trainingSkinListB))

for i in trainingNonSkinListB:
    nonskinAvgB+=int(i[1])
nonskinAvgB = int(nonskinAvgB/len(trainingNonSkinListB))

for i in trainingNonSkinListB:
    nonskinVarB += (int(i[1])-nonskinAvgB)**2
nonskinVarB = int(nonskinVarB/len(trainingNonSkinListB))

##########

for i in trainingSkinListR:
    skinAvgR+=int(i[1])
skinAvgR = int(skinAvgR/len(trainingSkinListR))

for i in trainingSkinListR:
    skinVarR += (int(i[1])-skinAvgR)**2
skinVarR = int(skinVarR/len(trainingSkinListR))

for i in trainingNonSkinListR:
    nonskinAvgR+=int(i[1])
nonskinAvgR = int(nonskinAvgR/len(trainingNonSkinListR))

for i in trainingNonSkinListR:
    nonskinVarR += (int(i[1])-nonskinAvgR)**2
nonskinVarR = int(nonskinVarR/len(trainingNonSkinListR))

##########

for i in trainingSkinListG:
    skinAvgG+=int(i[1])
skinAvgG = int(skinAvgG/len(trainingSkinListG))

for i in trainingSkinListG:
    skinVarG += (int(i[1])-skinAvgG)**2
skinVarG = int(skinVarG/len(trainingSkinListG))

for i in trainingNonSkinListG:
    nonskinAvgG+=int(i[1])
nonskinAvgG = int(nonskinAvgG/len(trainingNonSkinListG))

for i in trainingNonSkinListG:
    nonskinVarG += (int(i[1])-nonskinAvgG)**2
nonskinVarG = int(nonskinVarG/len(trainingNonSkinListG))

##########




def likelihood(x, mu,cov):
    part1 = 1 / ((np.power(2 * np.pi, 1.5)) * (np.power(np.linalg.det(cov), 0.5)))
    part2 = (-0.5) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(part1 * np.exp(part2))

def prior():
    pixList = []
    pr=0
    pg=0
    pb=0

    for i in range(0, 256):
        pixList.append(0)
    for skin in trainingSkinListR:
        pixList[int(skin[1])] += 1
    count = 0
    for pix in pixList:
        if pix != 0:
            count += 1
    pr = count / len(pixList)

    pixList=[]
    for i in range(0, 256):
        pixList.append(0)
    for skin in trainingSkinListG:
        pixList[int(skin[1])] += 1
    count = 0
    for pix in pixList:
        if pix != 0:
            count += 1
    pg = count / len(pixList)

    pixList = []
    for i in range(0, 256):
        pixList.append(0)
    for skin in trainingSkinListB:
        pixList[int(skin[1])] += 1
    count = 0
    for pix in pixList:
        if pix != 0:
            count += 1
    pb = count / len(pixList)

    return pr*pb*pg

def bayesian(tlikelihood,blikelihood, prior):
    return tlikelihood*prior/(tlikelihood*prior+blikelihood*(1-prior))

prior=prior()
print("prior:",prior)
#####################################################여기까지 수정했음
for i in range(0,len(trainingSkinListR)):
    cskinAvgListR.append(int(trainingSkinListR[i][1]))
    cskinAvgListG.append(int(trainingSkinListG[i][1]))
    cskinAvgListB.append(int(trainingSkinListB[i][1]))
    cnonskinAvgListR.append(int(trainingNonSkinListR[i][1]))
    cnonskinAvgListG.append(int(trainingNonSkinListG[i][1]))
    cnonskinAvgListB.append(int(trainingNonSkinListB[i][1]))

skinAvg = np.array([skinAvgR, skinAvgG, skinAvgB])
skinAvg.resize(3, 1)

nonskinAvg = np.array([nonskinAvgR, nonskinAvgG, nonskinAvgB])
nonskinAvg.resize(3, 1)

skinVar = np.vstack([cskinAvgListR, cskinAvgListG, cskinAvgListB])
skinVar = np.cov(skinVar)


nonskinVar = np.vstack([cnonskinAvgListR, cnonskinAvgListG, cnonskinAvgListB])
nonskinVar = np.cov(nonskinVar)

n = 0
for imgName in testSkinListR:
    n+=1
    print(imagepath+imgName[0].split('skin/')[1])
    img = cv2.imread(imagepath+imgName[0].split('skin/')[1], cv2.IMREAD_COLOR)
    rows, cols, channel = img.shape
    rmg = img[:, :, 2]
    gmg = img[:, :, 1]
    bmg = img[:, :, 0]
    height, width, channels = np.size(img, 0), np.size(img, 1), np.size(img, 2)
    #output = cv2.imread(imgName[0], cv2.IMREAD_GRAYSCALE)
    output = np.ones((height, width, 1), np.uint8)

    comparepath = skinpath + imgName[0].split('skin/')[1]
    compareImg = cv2.imread(comparepath, cv2.IMREAD_COLOR)

    for row in range(rows):
        for col in range(cols):
            r = rmg[row, col]
            g = gmg[row, col]
            b = bmg[row, col]
            k = np.array([r,g,b])
            l = compareImg[row, col]
            k.resize(3,1)

            if bayesian(likelihood(k,skinAvg,skinVar),likelihood(k,nonskinAvg,nonskinVar),prior)>=bayesian(likelihood(k,nonskinAvg,nonskinVar),likelihood(k,skinAvg,skinVar),(1-prior)):
                output[row,col]=255
                if l[0] != 0 and l[1] != 0 and l[2] != 0:
                    tpList.append("TP")
                else:
                    tpList.append("FN")

            else:
                output[row,col]=0
                if l[0] == 0 and l[1] == 0 and l[2] == 0:
                    tpList.append("FP")
                else:
                    tpList.append("TN")
    print("loop : ",n)

precision = tpList.count("TP") / (tpList.count("TP") + tpList.count("FP"))
recall = tpList.count("TP") / (tpList.count("TP") + tpList.count("FN"))
f = open(resultpath + "result.txt", 'a')
f.write("precision : " + str(precision) + "\n" + "recall : " + str(recall) + "\n")
f.close()

'''imgoutputpath = resultpath + imgName[0].split('skin/')[1].split('/img')[0]
    print(imgoutputpath)
    if not os.path.isdir(imgoutputpath):
        os.makedirs(os.path.join(imgoutputpath))
    cv2.imwrite(imgoutputpath.split('big/')[0]+imgName[0].split('/big')[1], output)
'''
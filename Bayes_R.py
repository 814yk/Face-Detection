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
resultpath=str(os.getcwd()).replace("\\","/")+"/venv/test/bayesian/R/"

gtFileName =[]

fileList = []
skinList = []

skinAvgList=[]
nonskinAvgList=[]
skinAvg=0
skinVar=0
nonskinAvg=0
nonskinVar=0
cnt = 1
prior = 0

tpList = []

for i in range(0,9):
    gtFileName.append("FDDB-fold-0"+str(i+1))
gtFileName.append("FDDB-fold-10")

for i in range(0,10):
    f = open(filepath + '/' + gtFileName[i] + '.txt', 'r', encoding="utf-8")
    lines = f.read().split()
    fileList.append(lines)
    f.close()

#########이미지 평균 계산################
'''for i in range(0,10):
    for j in range(0, len(fileList[i])):
        skinList.append(nonskinpath+fileList[i][j])
#print(skinList)

for imgName in skinList:
    tempSum = 0
    count = 0
    img = cv2.imread(imgName+".jpg", cv2.IMREAD_COLOR)
    rows, cols, channel = img.shape
    redImg = img[:, :, 2]
    for row in range(rows):
        for col in range(cols):
            k = redImg[row, col]
            if k != 0:
                tempSum += k
                count += 1
    skinAvgList.append(int(tempSum/count))
    f = open(outputpath+"nonskinAvg.txt",'a')
    f.write(imgName+".jpg"+','+str(int(tempSum/count))+'\n')
    f.close()
    print("loop : ",cnt)
    cnt+=1
'''

#print(skinAvgList)

f = open(outputpath+'skinAvgR.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    skinAvgList.append([skinpath+lines[i].split(',')[0]+'.jpg', lines[i].split(',')[1]])
print(skinAvgList)

f = open(outputpath+'nonskinAvgR.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    nonskinAvgList.append([nonskinpath+lines[i].split(',')[0]+'.jpg', lines[i].split(',')[1]])
print(nonskinAvgList)

#######################################
randIndex=[]
ran_num = random.randint(0,len(skinAvgList)-1)
for i in range(0,len(skinAvgList)):
    while ran_num in randIndex:
        ran_num = random.randint(0,len(skinAvgList)-1)
    randIndex.append(ran_num)

print(randIndex)
print(len(skinAvgList))
trainingSkinList = []
trainingNonSkinList = []
for i in range(0,1993):
    idx = randIndex[i]
    trainingSkinList.append(skinAvgList[idx])
    trainingNonSkinList.append(nonskinAvgList[randIndex[i]])
testSkinList = []
testNonSkinList = []
for i in range(1993, 2845):
    idx = randIndex[i]
    testSkinList.append(skinAvgList[idx])
    testNonSkinList.append(nonskinAvgList[randIndex[i]])
####################################

for i in trainingSkinList:
    skinAvg+=int(i[1])
skinAvg = int(skinAvg/len(trainingSkinList))

for i in trainingSkinList:
    skinVar += (int(i[1])-skinAvg)**2
skinVar = int(skinVar/len(trainingSkinList))

for i in trainingNonSkinList:
    nonskinAvg+=int(i[1])
nonskinAvg = int(nonskinAvg/len(trainingNonSkinList))

for i in trainingNonSkinList:
    nonskinVar += (int(i[1])-nonskinAvg)**2
nonskinVar = int(nonskinVar/len(trainingNonSkinList))

print(skinVar)
print(nonskinVar)

def likelihood(x, m,v):
    value = (1/(math.sqrt(2*math.pi)*math.sqrt(v))) * math.exp(((-1)*(x-m)**2)/(2*v))
    return value

def prior():
    pixList=[]
    for i in range(0,256):
        pixList.append(0)
    for skin in trainingSkinList:
        pixList[int(skin[1])]+=1
    count = 0
    for pix in pixList:
        if pix!=0:
            count+=1
    return count/len(pixList)

def bayesian(tlikelihood,blikelihood, prior):
    return tlikelihood*prior/(tlikelihood*prior+blikelihood*(1-prior))

prior=prior()
n = 0
for imgName in testSkinList:
    n+=1
    comparepath = skinpath+imgName[0].split('skin/')[1]
    img = cv2.imread(imagepath+imgName[0].split('skin/')[1], cv2.IMREAD_COLOR)
    compareImg = cv2.imread(comparepath, cv2.IMREAD_COLOR)
    rows, cols, channel = img.shape
    redImg = img[:, :, 2]
    height, width, channels = np.size(img, 0), np.size(img, 1), np.size(img, 2)
    #output = cv2.imread(imgName[0], cv2.IMREAD_GRAYSCALE)
    output = np.ones((height, width, 1), np.uint8)
    for row in range(rows):
        for col in range(cols):
            k = redImg[row, col]
            l = compareImg[row, col]
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
    #########이미지 출력################
    '''imgoutputpath = resultpath + imgName[0].split('skin/')[1].split('/img')[0]
    print(imgoutputpath)
    if not os.path.isdir(imgoutputpath):
        os.makedirs(os.path.join(imgoutputpath))
    cv2.imwrite(imgoutputpath.split('big/')[0]+imgName[0].split('/big')[1], output)'''

precision = tpList.count("TP")/(tpList.count("TP") + tpList.count("FP"))
recall = tpList.count("TP")/(tpList.count("TP") + tpList.count("FN"))
f = open(resultpath+"result.txt",'a')
f.write("precision : "+str(precision)+"\n"+"recall : "+str(recall)+"\n")

#f.write("precision : ", precision ,"\n","recall : ",recall ,"\n")
f.close()

print("precision : "+str(precision)+"\n"+"recall : "+str(recall)+"\n")
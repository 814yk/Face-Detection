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
resultpath=str(os.getcwd()).replace("\\","/")+"/venv/test/gaussian/RGB/"


gtFileName =[]

fileList = []
skinList = []
skinAvgList = []
nonskinAvgList = []
skinAvgListR=[]
nonskinAvgListR=[]
skinAvgListG=[]
nonskinAvgListG=[]
skinAvgListB=[]
nonskinAvgListB=[]
skinAvg=0
skinVar=0
nonskinAvg=0
nonskinVar=0
cnt = 1
prior = 0

tpList = []

'''for i in range(0,9):
    gtFileName.append("FDDB-fold-0"+str(i+1))
gtFileName.append("FDDB-fold-10")

for i in range(0,10):
    f = open(filepath + '/' + gtFileName[i] + '.txt', 'r', encoding="utf-8")
    lines = f.read().split()
    fileList.append(lines)
    f.close()
'''
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
    skinAvgListR.append([skinpath+lines[i].split(',')[0]+'.jpg', lines[i].split(',')[1]])

f = open(outputpath+'nonskinAvgR.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    nonskinAvgListR.append([nonskinpath+lines[i].split(',')[0]+'.jpg', lines[i].split(',')[1]])



f = open(outputpath+'skinAvgG.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    skinAvgListG.append([skinpath+lines[i].split(',')[0]+'.jpg', lines[i].split(',')[1]])

f = open(outputpath+'nonskinAvgG.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    nonskinAvgListG.append([nonskinpath+lines[i].split(',')[0]+'.jpg', lines[i].split(',')[1]])



f = open(outputpath+'skinAvgB.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    skinAvgListB.append([skinpath+lines[i].split(',')[0]+'.jpg', lines[i].split(',')[1]])

f = open(outputpath+'nonskinAvgB.txt', 'r', encoding="utf-8")
lines = f.read().split()
f.close()
for i in range(0, len(lines)):
    lines[i].split(',')
    nonskinAvgListB.append([nonskinpath+lines[i].split(',')[0]+'.jpg', lines[i].split(',')[1]])
print(skinAvgListR)

skinAvgList = skinAvgListR
nonskinAvgList = nonskinAvgListR
for i in range(0, len(skinAvgList)):
    skinAvgList[i][1]=(int(np.average([int(skinAvgListR[i][1]), int(skinAvgListG[i][1]), int(skinAvgListB[i][1])])))

for i in range(0, len(nonskinAvgListR)):
    nonskinAvgList[i][1]=(int(np.average([int(nonskinAvgListR[i][1]), int(nonskinAvgListG[i][1]), int(nonskinAvgListB[i][1])])))

print(skinAvgList)
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


def likelihood(x, m,v):
    value = (1/(math.sqrt(2*math.pi)*math.sqrt(v))) * math.exp(((-1)*(x-m)**2)/(2*v))
    return value

#############k-means##################
centroids = {}
classes = {}

def fit(data):
    for i in range(3):
       centroids[i]= int(data[i][1])
    for i in range(500):
        for i in range(3):
            classes[i] = []
        for features in data:
            distances = [np.linalg.norm(int(features[1]) - centroids[centroid]) for centroid in centroids]
            classification = distances.index(min(distances))
            classes[classification].append(features)

        previous = dict(centroids)

        for classification in classes:
            avg = 0
            centroids[classification] = 0
            for val in classes[classification]:
                avg += int(val[1])
            avg = avg / len(classes[classification])
            centroids[classification] = avg
        isOptimal = True

        for centroid in centroids:

            original_centroid = previous[centroid]
            curr = centroids[centroid]

            if np.sum((curr - original_centroid) / original_centroid * 100.0) > 0.0001:
                isOptimal = False
        if isOptimal:
            break

def predict(data):
    distances = [np.linalg.norm(data - centroids[centroid]) for centroid in  centroids]
    classification = distances.index(min(distances))
    return classification
#############k-means##################

for i in range(0, len(skinAvgList)):
    skinAvgList[i].append(i)
print(trainingSkinList)
fit(trainingSkinList)
print("class:",classes)

smu = []
svar = []
sm1, sm2, sm3 = 0, 0, 0
sv1, sv2, sv3 = 0, 0, 0
print(classes[0])

for cs in classes[0]:
    sm1 += int(cs[1])
sm1 = sm1/len(classes[0])

for cs in classes[1]:
    sm2 += int(cs[1])
sm2 = sm2/len(classes[1])

for cs in classes[2]:
    sm3 += int(cs[1])
sm3 = sm3/len(classes[2])

smu = [sm1, sm2, sm3]


for cs in classes[0]:
    sv1 += (int(cs[1])-sm1)**2
sv1 = sv1/len(classes[0])

for cs in classes[1]:
    sv2 += (int(cs[1])-sm1)**2
sv2 = sv2/len(classes[1])

for cs in classes[2]:
    sv3 += (int(cs[1])-sm1)**2
sv3 = sv3/len(classes[2])


svar = [sv1, sv2, sv3]

nonskinlist1 = []
nonskinlist2 = []
nonskinlist3 = []

for cs in classes[0]:
    num = cs[2]
    nonskinlist1.append(nonskinAvgList[num])

for cs in classes[1]:
    num = cs[2]
    nonskinlist2.append(nonskinAvgList[num])

for cs in classes[2]:
    num = cs[2]
    nonskinlist3.append(nonskinAvgList[num])


nsmu = []
nsvar = []
nsm1, nsm2, nsm3 = 0, 0, 0
nsv1, nsv2, nsv3 = 0, 0, 0

for ns in nonskinlist1:
    nsm1 += int(ns[1])
nsm1 = nsm1/len(nonskinlist1)

for ns in nonskinlist2:
    nsm2 += int(ns[1])
nsm2 = nsm2/len(nonskinlist2)

for ns in nonskinlist3:
    nsm3 += int(ns[1])
nsm3 = nsm3/len(nonskinlist3)

nsmu = [nsm1, nsm2, nsm3]

for ns in nonskinlist1:
    nsv1 += (int(ns[1])-nsm1)**2
nsv1 = nsv1/len(nonskinlist1)

for ns in nonskinlist2:
    nsv2 += (int(ns[1])-nsm2)**2
nsv2 = nsv2/len(nonskinlist2)

for ns in nonskinlist3:
    nsv3 += (int(ns[1])-nsm3)**2
nsv3 = nsv3/len(nonskinlist3)

nsvar = [nsv1, nsv2, nsv3]


n = 0
for imgName in testSkinList:
    n+=1
    comparepath = skinpath+imgName[0].split('skin/')[1]
    img = cv2.imread(imagepath+imgName[0].split('skin/')[1], cv2.IMREAD_COLOR)
    compareImg = cv2.imread(comparepath, cv2.IMREAD_COLOR)
    rows, cols, channel = img.shape
    blueImg = img[:, :, 0]
    greenImg = img[:, :, 1]
    redImg = img[:, :, 2]
    height, width, channels = np.size(img, 0), np.size(img, 1), np.size(img, 2)

    output = np.ones((height, width, 1), np.uint8)
    for row in range(rows):
        for col in range(cols):
            k = int(np.average([redImg[row, col], greenImg[row, col], blueImg[row, col]]))
            l = compareImg[row, col]
            if likelihood(k, smu[predict(k)], svar[predict(k)]) >= likelihood(k, nsmu[predict(k)], nsvar[predict(k)]):
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
   # imgoutputpath = resultpath + imgName[0].split('skin/')[1].split('/img')[0]
  #  print(imgoutputpath)
   # if not os.path.isdir(imgoutputpath):
  #      os.makedirs(os.path.join(imgoutputpath))
   # cv2.imwrite(imgoutputpath.split('big/')[0]+imgName[0].split('/big')[1], output)

precision = tpList.count("TP")/(tpList.count("TP") + tpList.count("FP"))
recall = tpList.count("TP")/(tpList.count("TP") + tpList.count("FN"))
f = open(resultpath+"result.txt",'a')
f.write("precision : "+str(precision)+"\n"+"recall : "+str(recall)+"\n")

#f.write("precision : ", precision ,"\n","recall : ",recall ,"\n")
f.close()

print("precision : "+str(precision)+"\n"+"recall : "+str(recall)+"\n")
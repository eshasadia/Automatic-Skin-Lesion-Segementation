import cv2 as cv
import  numpy as np
import  matplotlib.pyplot as plt
import csv

img = cv.imread('D:\Semester6\dataset\hrainx\IMD002.bmp',0)
# gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

#step-1 intensity contrast enhancement
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img = clahe.apply(img)


ret, thresh = cv.threshold(img, 0, 255,   cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# cv.imshow('image', thresh)

kernel = np.ones((3, 3), np.uint8)
median=cv.medianBlur(thresh,15)
closing = cv.morphologyEx(median, cv.MORPH_CLOSE, kernel, iterations=2)
opening=closing
cv.imshow('image', opening)
plt.imshow(img)
plt.show()

thresh1=cv.imread('D:\Semester6\dataset\hrainy\IMD002_lesion.bmp',0)


size=np.shape(thresh1)
r = size[0]
c = size[1]
print(r)
print(c)
tp = 0
fp = 0
tn = 0
fn = 0
for i in range(r):
    for j in range(c):
            if((thresh1[i][j]==255) and (opening[i][j]==255)):
                # img_bgr[i][j]=[0,0,0]
                tp += 1
            elif( (opening[i][j]==255) and (thresh1[i][j]!=opening[i][j])):
                # img_bgr[i][j] = [0, 0, 255]
                fp += 1
            elif ((thresh1[i][j]==0) and (opening[i][j]==0)):
                # img_bgr[i][j]= [255, 255, 255]
                tn += 1
            elif ((opening[i][j]==0) and (thresh1[i][j]!=opening[i][j])):
                # img_bgr[i][j] = [0, 255, 0]
                fn += 1

accuracy=(tn+tp)/(fn+fp+1+tn+tp)
dice_coeficient=(2*tp)/(fn+fp+(2*tp))
print("Accuracy = ",float((tn+tp)/(fn+fp+1+tn+tp)))
print("Dice Coefficient = ",float((2*tp)/(fn+fp+(2*tp))))
myFile = open('D:\Semester6\projects_labels.csv', 'a',newline='')
output=[['accuracy',accuracy],['dice_coeficient',dice_coeficient]]
with myFile:
   writer = csv.writer(myFile)
   writer.writerows(output)
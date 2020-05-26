import cv2
import numpy as np
import copy
import os

from cityscapesLabelHelper import color2label, labels, Label, id2label
relevantLabels = ['unlabeled', 'road', 'car', 'person', 'bicycle', 'bus', 'truck', 'lane']


def covertColorToLabelOrig(colorImg):
    labelImg = np.zeros(colorImg.shape)
    # color2label = lambda t: t ** 2
    # vfunc = np.vectorize(color2label)
    # labelImg = vfunc(colorImg)
    # labelImg = color2label[colorImg[:,:,:]].id

    for i in range(colorImg.shape[0]):
        for j in range(colorImg.shape[1]):
            pixelColor = (colorImg[i,j][0],colorImg[i,j][1],colorImg[i,j][2])
            labelImg[i,j] = color2label[pixelColor].id

    return labelImg



def covertColorToLabel(colorImg):
    # labelImg = np.zeros(colorImg.shape)
    #
    # for curLabel in labels:
    #     if curLabel.name in relevantLabels:
    #         # indicies = colorImg == list(curLabel.color)
    #         indicies = (colorImg[:,:] == list(curLabel.color)).all(axis=2)
    #
    #         labelImg[indicies] = curLabel.id
    #
    # return labelImg
    labelImg = np.zeros((colorImg.shape[0],colorImg.shape[1]))

    for curLabel in labels:
        if curLabel.name in relevantLabels:
            # indicies = colorImg == list(curLabel.color)
            indicies = (colorImg[:, :] == list(curLabel.color)).all(axis=2)

            labelImg[indicies] = curLabel.id

    return labelImg

def covertLabelToTrainId(labelImg):
    trainIDImg = np.zeros(labelImg.shape)
    for i in range(labelImg.shape[0]):
        for j in range(labelImg.shape[1]):
            pixelColor = labelImg[i,j][0]
            trainIDImg[i,j] = id2label[pixelColor].trainId
    return trainIDImg



def addInstanceBoundry(instanceImg, labelImg, boundryID=250):
    dx, dy = np.gradient(instanceImg)
    grad = (np.logical_or(dx,dy))

    #twice for thicker boundry
    dx, dy = np.gradient(grad)
    grad2 = (np.logical_or(dx,dy))
    grad = (np.logical_or(grad,grad2))


    labelImg[grad] = boundryID
    

    return labelImg

def convert(colorImageDir,saveLabelDir):
    counter = 0 
    addInstances = 0

    if not os.path.exists(saveLabelDir):
        os.makedirs(saveLabelDir)

    for root, dirs, files in os.walk(colorImageDir):
        extension = ".png"
        for file in files:
            if not extension in file:
                continue
            if "_instance" in file:
                continue
            if os.path.isfile(os.path.join(saveLabelDir, file)):
                print "Already Converted"
                continue

            colorImg = cv2.imread(os.path.join(root,file))
            colorImg =  cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)

            labelImg = covertColorToLabel(colorImg)

            if addInstances:
                if os.path.isfile(os.path.join(root,file.replace(".png","_instance.png"))):
                    instanceImg = cv2.imread(os.path.join(root,file.replace(".png","_instance.png")), cv2.IMREAD_UNCHANGED)
                    labelImg = addInstanceBoundry(instanceImg, labelImg)

            cv2.imwrite(os.path.join(saveLabelDir, file), labelImg) 
            counter += 1
            if counter % 50 == 0:
                print counter

if __name__ == '__main__':
    colorImageDir = "/media/artur/Data_ssd/01_Project/BSD_Camera_V2/BSD2V1/BSDdata/Origin/gtFine_Orign"
    saveLabelDir = "/media/artur/Data_ssd/01_Project/BSD_Camera_V2/BSD2V1/BSDdata/Origin/gtFine_simple"
    convert(colorImageDir,saveLabelDir)

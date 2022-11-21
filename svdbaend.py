import math

import matplotlib.image as image
import numpy.linalg as eigens
import numpy as np
import cv2 as cv
from PIL import Image as im
from math import isclose
import concurrent.futures
import time
from kivy.graphics.texture import Texture
import os
import skvideo.io
from itertools import repeat

def SVD(img,rankk=None,rankOpt='full',percentage=None):
    #img = np.array([[1,0,3],[9,0,1],[0,1,1],[1,10,5],[12,61,977]])
    mode = ''
    if np.argmin(np.asarray(img.shape).flatten()) == 0:
        mode = 'U'
        img = np.transpose(img)
    else:
        mode = 'V' #imgimgT result in V
    imgTimg = np.dot(np.transpose(img), img)
    a = imgTimg
    eigArr,eigVec = eigens.eig(a) #Calculate eigenvalue and eigenVector eigArr nx1 eigVec mxn
    eigArr = np.abs(eigArr) #eigen = singularvalue ^2 => singularvalue
    i = eigArr.argsort()[::-1] #[::-1] reverse the sort order,argsort return index array
    eigArr = eigArr[i]
    #rank = int(np.array(np.where(eigArr>0)[0])[-2:-1:1]) #Take last index where eigArr > 0
    # if len(np.array(np.where(eigArr > 0)[0])) == 0:
    #     rank = int(np.array(np.where(eigArr == 0)[0])[0])
    # else:
    #     rank = int(np.array(np.where(eigArr > 0)[0])[-2:-1:1])
    #smallestPositiveEign = min(a for a in eigArr[::-1] if a > 0) #Gia tri rieng nho nhat A^TA (nxn)=> rank <=n
    #eigArr = np.real(eigArr)  #Due to finite precision, there might be complex numbers
    # lenEig = len(eigArr)
    # for e in eigArr[::-1]:
    #     if math.isclose(smallestPositiveEign,e,rel_tol=0.0001):
    #         rank = lenEig
    #         break
    #     else:
    #         lenEig-=1

    if rankOpt == 'usr' and rankk != None and rankk <= len(eigArr):
        rank = rankk
    else:
        rank = len(eigArr)
    if rankOpt == 'opt' and percentage != None:
        rankApprox = len(eigArr)
        SumeigSquare = sum(eigArr)
        for o in range(1,len(eigArr)):
            num = sum(eigArr[:o])/SumeigSquare
            if num >= percentage:
                rankApprox = o
                break
        print(rankApprox)
        rank = rankApprox

    eigArr = np.sqrt(eigArr[:rank]) #rank x 1

    eigVec = eigVec[:,i]
    #eigVec = eigVec[:,:rank]
    eigVec = np.real(eigVec) #Due to finite precision, there might be complex numbers, cast from i to real numbers

    sigma = np.diag(eigArr) #Turn eigArr to matrix with eigArr on its diagonal rank x rank
    sigma = np.pad(sigma,((0,0),(0,img.shape[1]-sigma.shape[1])),mode='constant',constant_values=0) #Fill sigma with zeros so the matrix has size rankxn
    U = np.empty(shape=(img.shape[0],rank)) #Intilize matrix U with size mxrank
    U.fill(0)
    V = eigVec

    # for i in range(0,rank):
    #     U[:,i] = (1/eigArr[i])*np.dot(img,V[:,i])

    eigArrclone = np.array([eigArr,]*img.shape[0]) #Turn size rankx1 to rankxm by cloning by its column
    U = np.divide(np.dot(img,V[:,:rank]),eigArrclone)
    imgg = np.dot(np.dot(U, sigma), np.transpose(V)) #U x sigma x V (mxm mxn x nxn) => mxn (mxrank rankxn nxn) => mxn

    if mode == 'U':
        imgg = np.transpose(imgg)
    imgg = np.real(imgg)
    #print(imgg)
    imgg = np.where(imgg<0,0,imgg)
    data = im.fromarray(imgg).convert('L')


    return data

#SVD(np.array(cv.cvtColor(cv.imread('Example-of-denoising-results-of-a-part-of-the-image-Cameraman-a-Noisy-image-Peak5.png'),code=cv.COLOR_BGR2GRAY)))
# imgMatrix = np.array(cv.imread('a.png'))
# imgMatrixB = imgMatrix[:,:,0]
# imgMatrixG = imgMatrix[:,:,1]
# imgMatrixR = imgMatrix[:,:,2]
# bChannel = SVD(imgMatrixB)
# gChannel = SVD(imgMatrixG)
# rChannel = SVD(imgMatrixR)
# #
# rgb = np.dstack((bChannel,gChannel,rChannel))
# print(rgb.shape)
# image = im.fromarray(rgb).convert('RGB')
# cv.imshow('RGB',np.array(image))
# cv.waitKey(0)
def imagesvd(img,rankk=None,rankOpt='usr',mode='rgb',percentage=None,forVid=True,multiThreaded=True):
    percentage=.95
    if forVid == False:
        imgMatrix = np.array(cv.imread(img))
    else:
        imgMatrix = np.array(img)
    if mode == 'rgb' or mode == 'grb' or mode == 'bgr':
        imgMatrixB = imgMatrix[:,:,0]
        imgMatrixG = imgMatrix[:,:,1]
        imgMatrixR = imgMatrix[:,:,2]
        if multiThreaded == False:
            bChannel = SVD(imgMatrixB,rankk=rankk,rankOpt=rankOpt)
            gChannel = SVD(imgMatrixG,rankk=rankk,rankOpt=rankOpt)
            rChannel = SVD(imgMatrixR,rankk=rankk,rankOpt=rankOpt)
            if mode == 'rgb': imgStack = np.dstack((bChannel,gChannel,rChannel)) #bgr
            if mode == 'grb': imgStack = np.dstack((gChannel,rChannel,bChannel))
            if mode == 'bgr': imgStack = np.dstack((rChannel,gChannel,bChannel))
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futureB = executor.submit(SVD, imgMatrixB,rankk=rankk,rankOpt=rankOpt)
                futureG = executor.submit(SVD, imgMatrixG,rankk=rankk,rankOpt=rankOpt)
                futureR = executor.submit(SVD, imgMatrixR,rankk=rankk,rankOpt=rankOpt)
            if mode == 'rgb': imgStack = np.dstack((futureB.result(), futureG.result(), futureR.result()))
            if mode == 'grb': imgStack = np.dstack((futureG.result(),futureR.result(),futureB.result()))
            if mode == 'bgr': imgStack = np.dstack((futureR.result(),futureG.result(),futureB.result()))
        image = im.fromarray(imgStack).convert('RGB')
        if forVid == False:
            cv.imshow('Image',np.array(image))
            cv.waitKey(0)
    if mode == 'gs':
        image = SVD(np.array(cv.cvtColor(imgMatrix,code=cv.COLOR_BGR2GRAY)),rankk,rankOpt,percentage)
        if forVid == False:
            cv.imshow('GrayScale', np.array(image))
    if forVid == False: cv.imwrite('result.png', np.array(image))
    return image

def videosvd(vid,rankk=None,rankOpt='full',mode='rgb',multiThreaded=True):
    readFps = cv.VideoCapture(vid)
    fps = readFps.get(cv.CAP_PROP_FPS)
    print(fps)
    print(rankk)
    vid = cv.VideoCapture(vid)
    orgimg_arr=[]
    while True:
        success,frame = vid.read()
        if not success:
            cv.destroyAllWindows()
            break
        orgimg_arr.append(frame)
    img_arr = []
    # with concurrent.futures.ThreadPoolExecutor(5) as executor:
    #     for frame in executor.map(imagesvd,orgimg_arr[0:len(orgimg_arr[0])],repeat(rankk),repeat(rankOpt),repeat(mode),repeat(multiThreaded)):
    #         print(len(img_arr))
    #         img_arr.append(np.array(frame))
    with concurrent.futures.ThreadPoolExecutor(5) as executor:
        img_arr = list(executor.map(imagesvd,orgimg_arr[0:len(orgimg_arr[0])],repeat(rankk),repeat(rankOpt),repeat(mode),repeat(multiThreaded)))
    for frame in img_arr:
        cv.imshow('video',np.array(frame))
        cv.waitKey(50)
    out = cv.VideoWriter('resultVid.mp4',cv.VideoWriter_fourcc(*'mp4v'),fps,img_arr[0].size)
    for i in range(len(img_arr)):
        out.write(np.array(img_arr[i]))
    out.release()

def webcamsvd(rankk,rankOpt,mode):
    cap = cv.VideoCapture(0) #Index 0 webcam
    i=0
    while True:
        i+=1
        if cv.waitKey(1) == ord('q'):
            break
        success,frame = cap.read()
        if not success:
            break
        frame = cv.resize(frame,dsize=None,fx=0.6,fy=0.6)
        imgMatrix = np.array(frame)
        imgMatrixB = imgMatrix[:, :, 0]
        imgMatrixG = imgMatrix[:, :, 1]
        imgMatrixR = imgMatrix[:, :, 2]
        start = time.time()

        #MultiThreading
        # try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futureB = executor.submit(SVD,imgMatrixB)
            futureG = executor.submit(SVD,imgMatrixG)
            futureR = executor.submit(SVD,imgMatrixR)
        rgb = np.dstack((futureB.result(),futureG.result(),futureR.result()))
        # except:
        #     print()

        #Single thread
        # bChannel = SVD(imgMatrixB)
        # gChannel = SVD(imgMatrixG)
        # rChannel = SVD(imgMatrixR)
        # rgb = np.dstack((bChannel,gChannel,rChannel))


        image = im.fromarray(rgb).convert('RGB')
        end = time.time()
        seconds = end - start

        font = cv.FONT_HERSHEY_PLAIN
        org = (10, 50)
        fontScale = 2
        color = (238, 75, 43)
        thickness = 2
        imm = np.array(image)
        cv.putText(imm,str(1/seconds), org, font, fontScale, color, thickness, cv.LINE_AA)
        cv.imshow('RGB', imm)

    cap.release()
    cv.destroyAllWindows()









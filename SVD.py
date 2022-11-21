import math

import matplotlib.image as image
import numpy.linalg as eigens
import numpy as np
import cv2 as cv
from PIL import Image as im
from math import isclose
import concurrent.futures
import time

def SVD(img):
    #img = np.array([[1,0,1],[-2,1,0]])
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
    smallestPositiveEign = min(a for a in eigArr[::-1] if a > 0) #Gia tri rieng nho nhat A^TA (nxn)=> rank <=n
    eigArr = np.real(eigArr)  #Due to finite precision, there might be complex numbers  
    # lenEig = len(eigArr)
    # for e in eigArr[::-1]:
    #     if math.isclose(smallestPositiveEign,e,rel_tol=0.0001):
    #         rank = lenEig
    #         break
    #     else:
    #         lenEig-=1
    rank = len(eigArr) -10

    # rankApprox = len(eigArr)
    # SumeigSquare = sum(np.square(eigArr))
    # for o in range(1,len(eigArr)):
    #     num = sum(np.square(eigArr[:o]))/SumeigSquare
    #     if num >= 0.95:
    #         rankApprox = o
    #         break
    # print(rankApprox)
    # rank = rankApprox

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

    #Single threading
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









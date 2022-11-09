import cv2 as cv
import pywt
import pywt.data
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt


def mean(a,b):
    return (a+b)/2

def max(a,b):
    x, y = a.shape
    c = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if a[i,j] > b[i,j]:
                c[i,j] = a[i,j]
            else:
                c[i,j] = b[i,j]
    return c


def min(a,b):
    x, y = a.shape
    c = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if a[i, j] < b[i, j]:
                c[i, j] = a[i, j]
            else:
                c[i, j] = b[i, j]
    return c





def fusion(i1,i2,mode):
    coe1 = pywt.dwt2(i1, "coif5", "periodization")
    cA1, (cH1, cV1, cD1) = coe1

    coe2 = pywt.dwt2(i2, "coif5", "periodization")
    cA2, (cH2, cV2, cD2) = coe2

    cA3 = cH3 = cV3 = cD3 = 0
    if mode == 0: #meanmean
        cA3 = mean(cA1, cA2)
        cH3 = mean(cH1, cH2)
        cV3 = mean(cV1, cV2)
        cD3 = mean(cD1, cD2)
    elif mode == 1: #meanmin
        cA3 = mean(cA1, cA2)
        cH3 = min(cH1, cH2)
        cV3 = min(cV1, cV2)
        cD3 = min(cD1, cD2)
    elif mode == 2: #meanmax
        cA3 = mean(cA1, cA2)
        cH3 = max(cH1, cH2)
        cV3 = max(cV1, cV2)
        cD3 = max(cD1, cD2)
    elif mode == 3: #minmean
        cA3 = min(cA1, cA2)
        cH3 = min(cH1, cH2)
        cV3 = min(cV1, cV2)
        cD3 = mean(cD1, cD2)
    elif mode == 4: #minmin
        cA3 = min(cA1, cA2)
        cH3 = min(cH1, cH2)
        cV3 = min(cV1, cV2)
        cD3 = min(cD1, cD2)
    elif mode == 5: #minmax
        cA3 = min(cA1, cA2)
        cH3 = max(cH1, cH2)
        cV3 = max(cV1, cV2)
        cD3 = max(cD1, cD2)
    elif mode == 6: #maxmean
        cA3 = max(cA1, cA2)
        cH3 = mean(cH1, cH2)
        cV3 = mean(cV1, cV2)
        cD3 = mean(cD1, cD2)
    elif mode == 7: #maxmin
        cA3 = max(cA1, cA2)
        cH3 = min(cH1, cH2)
        cV3 = min(cV1, cV2)
        cD3 = min(cD1, cD2)
    elif mode == 8: #maxmax
        cA3 = max(cA1, cA2)
        cH3 = max(cH1, cH2)
        cV3 = max(cV1, cV2)
        cD3 = max(cD1, cD2)
    coef = cA3, (cH3, cV3, cD3)
    f = pywt.idwt2(coef,"coif5",mode="periodization")
    return f

i1 = cv.imread("Patient Data/p3/ct.jpg", 1)
i2 = cv.imread("Patient Data/p3/mri_registered.jpg", 1)
i2 = cv.resize(i2,(i1.shape[1], i1.shape[0]))
b1, g1, r1 = cv.split(i1)
b2, g2, r2 = cv.split(i2)

type = 0
fg = fusion(g1, g2, type)
fr = fusion(r1, r2, type)
fb = fusion(b1, b2, type)
fr = cv.resize(fr,(i1.shape[1], i1.shape[0]))
fg = cv.resize(fg,(i1.shape[1], i1.shape[0]))
fb = cv.resize(fb,(i1.shape[1], i1.shape[0]))


f = i1.copy()
f[:,:,0] = fb[:,:]
f[:,:,1] = fg[:,:]
f[:,:,2] = fr[:,:]

f = np.multiply(np.divide(f - np.min(f),(np.max(f) - np.min(f))),255)
f = f.astype(np.uint8)
cv.imshow("f",f)
cv.waitKey(0)
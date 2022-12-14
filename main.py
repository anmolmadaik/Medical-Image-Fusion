import cv2 as cv
import pywt
import pywt.data
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import pyshearlab as psl

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


def lowfusion(i1,i2,mode):
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

def highfusion(im1, im2):
    shearSys1 = psl.SLgetShearletSystem2D(0, im1.shape[0], im1.shape[1], 3)
    coeff1 = psl.SLsheardec2D(im1, shearSys1)
    coeff1 = np.real(coeff1)

    shearSys2 = psl.SLgetShearletSystem2D(0, im2.shape[0], im2.shape[1], 3)
    coeff2 = psl.SLsheardec2D(im2, shearSys2)
    coeff2 = np.real(coeff2)
    coefffused = (coeff1 + coeff2)/2
    imrec = psl.SLshearrec2D(coefffused,shearSys1)
    return imrec

def printPSNR(i1,i2,fl,fh,sb):
    i1.astype(np.uint8)
    i2.astype(np.uint8)
    fl.astype(np.uint8)
    fh.astype(np.uint8)
    sb.astype(np.uint8)
    a = cv.PSNR(fl, i1)
    b = cv.PSNR(fh, i1)
    c = cv.PSNR(sb, i1)
    print(a,b,c,end="\n")

i1 = cv.imread("Patient Data/p3/ct.jpg", 1)
i2 = cv.imread("Patient Data/p3/mri_registered.jpg", 1)
i2 = cv.resize(i2,(i1.shape[1], i1.shape[0]))
b1, g1, r1 = cv.split(i1)
b2, g2, r2 = cv.split(i2)


type = 0
flg = lowfusion(g1, g2, type)
flr = lowfusion(r1, r2, type)
flb = lowfusion(b1, b2, type)
flr = cv.resize(flr,(i1.shape[1], i1.shape[0]))
flg = cv.resize(flg,(i1.shape[1], i1.shape[0]))
flb = cv.resize(flb,(i1.shape[1], i1.shape[0]))


fl = i1.copy()
fl[:,:,0] = flb[:,:]
fl[:,:,1] = flg[:,:]
fl[:,:,2] = flr[:,:]



fl = np.multiply(np.divide(fl - np.min(fl),(np.max(fl) - np.min(fl))),255)
fl = fl.astype(np.uint8)


b1 = b1.astype(float)
g1 = g1.astype(float)
r1 = r1.astype(float)
b2 = b2.astype(float)
g2 = g2.astype(float)
r2 = r2.astype(float)

fhg = highfusion(g1, g2)
fhr = highfusion(r1, r2)
fhb = highfusion(b1, b2)

fh = i1.copy()
fh[:,:,0] = fhb[:,:]
fh[:,:,1] = fhg[:,:]
fh[:,:,2] = fhr[:,:]

fh = np.multiply(np.divide(fh - np.min(fh),(np.max(fh) - np.min(fh))),255)
fh = fh.astype(np.uint8)

superimposed = cv.add(i1,i2)

cv.imshow("CT Scan:", i1)
cv.imshow("MRI Scan:", i2)
cv.imshow("Low Frequency Fusion: ", fl)
cv.imshow("High Frequency Fusion: ", fh)
cv.imshow("Superimposing: ", superimposed)

printPSNR(i1, i2, fl, fh, superimposed)

cv.waitKey(0)
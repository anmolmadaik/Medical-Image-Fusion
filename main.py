import cv2 as cv
import pywt
import pywt.data
import numpy as np
import pyshearlab as psl
from skimage.metrics import structural_similarity
from PIL import Image

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

def weight(im): # calculated so that brighter parts overlap details having low brightness with a lesser extent
    x, y = im.shape
    count = 0
    sum = 0
    for i in range(x):                          # calculate average intensity of image, except black parts
        for j in range(y):
            b = im[i, j]
            if b != 0:
                count = count + 1
                sum = sum + b
    average = 255 - (sum/count)                 #since brighter images should have less weight, we can map it by inverting the intensity obtained
    return average


def fusion_wt(i1,i2,mode):                      #image fusion is done using wavelet transform
    w1 = weight(i1)
    w2 = weight(i2)
    total = w1 + w2
    r1 = float(w1/total)                        #weight of the first image based on its average intensity
    r2 = float(w2/total)                        #weight of the second image based on its average intensity

    coe1 = pywt.dwt2(i1, "coif5", "periodization") #convert image into wavelet coefficients
    cA1, (cH1, cV1, cD1) = coe1
    cA1 = r1 * cA1
    cH1 = r1 * cH1
    cV1 = r1 * cV1
    cD1 = r1 * cD1

    coe2 = pywt.dwt2(i2, "coif5", "periodization") #convert image into wavelet coefficients
    cA2, (cH2, cV2, cD2) = coe2
    cA2 = r2 * cA2
    cH2 = r2 * cH2
    cV2 = r2 * cV2
    cD2 = r2 * cD2

    cA3 = cH3 = cV3 = cD3 = 0
    if mode == 0: #meanmean   #combining coefficients generated of two images using different techniques
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
    f = pywt.idwt2(coef,"coif5",mode="periodization") #reconstructing the image based on combined coefficients to find the fused image
    return f

def fusion_nsst(im1, im2):
    shearSys1 = psl.SLgetShearletSystem2D(0, im1.shape[0], im1.shape[1], 3) #creates a shearlet system
    coeff1 = psl.SLsheardec2D(im1, shearSys1) #decomposes the image into shearlet coefficients associated with the shearlet system
    coeff1 = np.real(coeff1)
    shearSys2 = psl.SLgetShearletSystem2D(0, im2.shape[0], im2.shape[1], 3)
    coeff2 = psl.SLsheardec2D(im2, shearSys2)
    coeff2 = np.real(coeff2)
    w1 = weight(im1)
    w2 = weight(im2)
    total = w1 + w2
    r1 = float(w1/total)
    r2 = float(w2/total)
    coefffused = (r1*coeff1 + r2*coeff2)/2  #Add two coefficients using their mean
    imrec = psl.SLshearrec2D(coefffused,shearSys1) #reconstruct the image based on shearlet transform obtained
    return imrec


def mutual_information(image,fused):
    hist, x, y = np.histogram2d(image.ravel(), fused.ravel(), 256)
    pxy = hist/float(np.sum(hist))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:,None] * py[:None]
    nonzeros = pxy>0
    return np.sum(pxy[nonzeros] * np.log(pxy[nonzeros] / px_py[nonzeros]))

def performance_metrics(input1, input2, fused, case):
    if case == 0:
        method = "Discrete Wavelet Transform"
    elif case == 1:
        method = "Non-Subsampled Shearlet Transform"
    else:
        method = "Superimposing"
    #Calculate PSNR
    psnr1 = cv.PSNR(fused, input1)
    psnr2 = cv.PSNR(fused, input2)
    print("PSNR of fused image using " + method + "with respect to CT: ",psnr1)
    print("PSNR of fused image using " + method + "with respect to MRI: ", psnr2)

    #Calculate Fusion Factor
    m1 = mutual_information(input1, fused)
    m2 = mutual_information(input2, fused)
    ff = m1+m2
    print("Fusion Factor for " + method + ": ",ff)

    #Calculate SSIM
    i1 = cv.cvtColor(input1, cv.COLOR_BGR2GRAY)
    i2 = cv.cvtColor(input2, cv.COLOR_BGR2GRAY)
    f = cv.cvtColor(fused, cv.COLOR_BGR2GRAY)
    (score1, diff1) = structural_similarity(f, i1, full = True)
    (score2, diff2) = structural_similarity(f, i2, full = True)
    print("Structural Similarity Index of fused output using " + method + " with respect to CT: ", score1)
    print("Structural Similarity Index of fused output using " + method + " with respect to MRI: ", score2)


i1 = cv.imread("Patient Data/p9/ct.jpg", 1)  #read CT Scan image
i2 = cv.imread("Patient Data/p9/mri_registered.jpg", 1) #read MRI image
i2 = cv.resize(i2,(i1.shape[1], i1.shape[0]))
b1, g1, r1 = cv.split(i1)
b2, g2, r2 = cv.split(i2)


type = 0
flg = fusion_wt(g1, g2, type) #fusion done for R,G,B, planes separately using Discrete Wavelet Transform
flr = fusion_wt(r1, r2, type)
flb = fusion_wt(b1, b2, type)
flr = cv.resize(flr,(i1.shape[1], i1.shape[0]))
flg = cv.resize(flg,(i1.shape[1], i1.shape[0]))
flb = cv.resize(flb,(i1.shape[1], i1.shape[0]))


fl = i1.copy() #combining R,G,B, components into single image
fl[:,:,0] = flb[:,:]
fl[:,:,1] = flg[:,:]
fl[:,:,2] = flr[:,:]


fl = np.multiply(np.divide(fl - np.min(fl),(np.max(fl) - np.min(fl))),255) #Image Enhancement
fl = fl.astype(np.uint8)


b1 = b1.astype(float)
g1 = g1.astype(float)
r1 = r1.astype(float)
b2 = b2.astype(float)
g2 = g2.astype(float)
r2 = r2.astype(float)

fhg = fusion_nsst(g1, g2)  #fusion done for R,G,B, components separately using NSST
fhr = fusion_nsst(r1, r2)
fhb = fusion_nsst(b1, b2)

fh = i1.copy()
fh[:,:,0] = fhb[:,:]  #Combining R,G,B components into single image
fh[:,:,1] = fhg[:,:]
fh[:,:,2] = fhr[:,:]

fh = np.multiply(np.divide(fh - np.min(fh),(np.max(fh) - np.min(fh))),255) #image enhancement
fh = fh.astype(np.uint8)

su = cv.add(i1, i2) #Superimposing two images

cv.imshow("CT Scan:", i1)
cv.imshow("MRI Scan:", i2)
cv.imshow("Fusion using Discrete Wavelet Transform: ", fl)
cv.imshow("Fusion using Non-Subsampled Shearlet Transform: ", fh)
cv.imshow("Superimposing: ", su)

performance_metrics(i1, i2, fl, 0)
performance_metrics(i1, i2, fh, 1)
performance_metrics(i1, i2, su, 2)

cv.waitKey(0)
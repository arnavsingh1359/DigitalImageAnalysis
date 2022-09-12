import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import exposure
from scipy.signal import argrelextrema
import seaborn as sns
from scipy import stats
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.signal import correlate2d
from math import sqrt, pi
from scipy.signal import medfilt
from scipy import ndimage

img = np.array(cv.imread("Pictures/part1/img_1.png"))

# lum, alpha, beta = np.array_split(cv.cvtColor(img, cv.COLOR_BGR2LAB), 3, axis=2)
# hue, sat, value = np.array_split(cv.cvtColor(img, cv.COLOR_BGR2HSV), 3, axis=2)

# Implemented, Adaptive Luminance Enhancement from AINDANE


def detected_skin(image, alpha_a=6.5, beta_b=12, sat_min=15, sat_max=170, hue_max=17.1):
    lum,a,b = np.array_split(cv.cvtColor(image, cv.COLOR_BGR2Lab), 3, axis=2)
    hue, sat, value = np.array_split(cv.cvtColor(image, cv.COLOR_BGR2HSV), 3, axis=2)
    Y,Cr,Cb = np.array_split(cv.cvtColor(image,cv.COLOR_BGR2YCR_CB),3,axis =2)
    hue = hue[:, :, 0]
    sat = sat[:, :, 0]
    value = value[:, :, 0]
    Y = Y[:, :, 0]
    Cr = Cr[:, :, 0]
    Cb = Cb[:,:,0]
    lum = lum[:, :, 0]
    a = a[:, :, 0]
    b = b[:, :, 0]
    result = np.zeros(lum.shape,image.dtype)			
    p = 0
    #INITIAL TEST
    while p < result.shape[0]:
        q = 0
        while q < result.shape[1]:            
            if sat_min <= sat[p, q] <= sat_max and hue[p, q] <= hue_max :
                if 135<=Cr[p,q]<=180 and 85<=Cb[p,q]<=135:
                    result[p, q] = 1
            q += 1
        p += 1
    
    # SECOND TESTING
    while p < result.shape[0]:
        q = 0
        while q < result.shape[1]:            
            if (1.0*(a[p,q]-143)/alpha_a)**2 + (1.0*(b[p,q]-148)/beta_b)**2 <1.25 :
                if 16.25 <= sat[p, q] <= 191.25  & hue[p,q]<hue_max:
                    result[p, q] = 1
            q += 1
        p += 1

    #skin patching
    _h,_w = image.shape[:2]
    _kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(int(_h/48),int(_w/48)))
    skin_mask_closed = cv.morphologyEx(result,cv.MORPH_CLOSE,_kernel)
    
    return skin_mask_closed*255

def detect_faces(image, scale_factor=1.01, min_neighbor=6):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scale_factor, min_neighbor)
    return faces


def WLS_filter(img, lambda_ = 0.1,alpha = 1.2, eps = 10^-4):
    image = img/255.0

    s = image.shape

    k = np.prod(s)

    dy = np.diff(image, 1, 0)
    dy = -lambda_ / (np.absolute(dy) ** alpha + eps)
    last_y = np.zeros((s[1],))
    dy = np.vstack((dy, last_y))
    dy = dy.flatten('F')

    dx = np.diff(image, 1, 1)
    dx = -lambda_ / (np.absolute(dx) ** alpha + eps)
    last_x = np.zeros((s[0],))
    dx = np.hstack((dx, last_x[:, np.newaxis]))
    dx = dx.flatten('F')

    a = spdiags(np.vstack((dx, dy)), [-s[0], -1], k, k)

    d = 1 - (dx + np.roll(dx, s[0]) + dy + np.roll(dy, 1))
    a = a + a.T + spdiags(d, 0, k, k )
    _out = spsolve(a, image.flatten('F')).reshape(s[::-1])

    base  = np.rollaxis(_out,1)*255
    out = np.clip(base, 0, 255)
    detail = image - out
    np.clip(detail, 0, 255, out=detail)
    return out, detail

def eacp(F, I, W= None, lambda_=0.2, alpha=0.3, eps=1e-4):
    if W is None:
        W= np.ones(F.shape)
    L=np.log(I+eps)
    f= F.flatten('F')
    w = W.flatten('F')
    s = L.shape

    k = np.prod(s)
    # L_i - L_j along y axis
    dy = np.diff(L, 1, 0)
    dy = -lambda_ / (np.absolute(dy) ** alpha + eps)
    dy = np.vstack((dy, np.zeros(s[1], )))
    dy = dy.flatten('F')
    # L_i - L_j along x axis
    dx = np.diff(L, 1, 1)
    dx = -lambda_ / (np.absolute(dx) ** alpha + eps)
    dx = np.hstack((dx, np.zeros(s[0], )[:, np.newaxis]))
    dx = dx.flatten('F')
    # A case: j \in N_4(i)  (neighbors of diagonal line)
    a = spdiags(np.vstack((dx, dy)), [-s[0], -1], k, k)
    # A case: i=j   (diagonal line    # )
    d = w - (dx + np.roll(dx, s[0]) + dy + np.roll(dy, 1))
    a = a + a.T + spdiags(d, 0, k, k) # A: put together
    f = spsolve(a, w*f).reshape(s[::-1]) # slove Af  =  b =w*g and restore 2d
    A = np.rollaxis(f,1)
    # A = np.clip( _out*255.0, 0, 255).astype('uint8')
    return A

def is_bimodal(x, y):
    maxima = argrelextrema(y, np.greater)[0]
    minima = argrelextrema(y, np.less)[0]
    if len(maxima) >= 2:
        i = 0
        while i < len(maxima) - 1 and len(minima) > 0:
            d = y[maxima[i]]
            b = y[maxima[i + 1]]
            m = y[minima[i]]
            if m < 0.8 * d and m < 0.8 * b:
                return int(x[maxima[i]]), int(x[maxima[i + 1]]), int(x[minima[i]])
            i += 1

    return None, None, None

def mask_skin(img,mask):
    for i in range(len(mask[0])):
        for j in range(len(mask[1])):
            if mask[i][j] == 0:
                img[i,j] =0
    return img

def percentile(image):
    data = np.zeros(256)
    for i in range(len(image[0])):
        for j in range(len(image[1])):
            data[int(image[i,j])] +=1
    
    sum = 0
    total = int(np.prod(image.shape)*0.75)
    cum = np.zeros(256)
    for i in range(len(data)):
        sum += data[i]
        cum[i] = sum
        if cum[i]>=total:
            return i
            
def allzero(img):
    h,j = img.shape
    for i in range(h):
        for j in range(j):
            if img[i][j] != 0 :
                return False
    return True


def face_correction(image,skin_mask , lambda_=0.2):

    lum, alpha, beta = np.array_split(cv.cvtColor(image, cv.COLOR_BGR2Lab), 3, axis=2)
    lum = lum[:, :, 0]
    alpha = alpha[:, :, 0]
    beta = beta[:, :, 0]
    I_out,detail = WLS_filter(lum)
    cv.imshow('mask',skin_mask)
    cv.waitKey(0)
    #face_detection
    face = detect_faces(image, 1.001, 5)
    #all faces
    faces = [lum[y:y + h, x:x + w] for (x, y, w, h) in face]
    W = np.zeros(lum.shape)
    A = np.ones(lum.shape, dtype="float")
    B = np.ones(lum.shape)
    # ##face_correction a)sidelight and b)exposure
    for index,(x, y, w, h) in enumerate(face):
        #sidelight correction
        skin_in_face = skin_mask[y:y+h, x:x+w]
        data, bins = exposure.histogram(lum[y:y + h, x:x + w], nbins=256)
        # plt.hist(data, bins)
        intensity = np.mgrid[0:256]
        kernel = stats.gaussian_kde(data)
        density = kernel(intensity)
        # plt.plot(intensity, density)
        maxima = argrelextrema(density, np.greater)
        minima = argrelextrema(density, np.less)
        d, b, m = is_bimodal(intensity, density)     
        #global W
        if d and b and m:
            f = (b - d) / (m - d)
            r = 0
            while r < w:
                s = 0
                while s < h:
                    if skin_in_face[r, s] and (lum[y + r, x + s] < m):
                        A[y + r, x + s] = f
                    s += 1
                r += 1
            miu = (lum[A == f]).mean()
            sig = 255 * 3  # manually set, 3*sig = 120 close to 255/2
            W = np.exp(-(lum - miu) ** 2 / sig ** 2)
            W[...] = 1 - W[...]
            W[...] = 1  
    are_all_zero = allzero(W.copy())
    if are_all_zero:
        W = np.ones(lum.shape)
        A_after = eacp(A,lum,W, lambda_)
    else:
        A_after =eacp(A,lum,W,lambda_)
    I_out *=A_after
    I_out2 = I_out.copy()
    for index,(x, y, w, h) in enumerate(face):
        
        skin_in_face = skin_mask[y:y+h, x:x+w]    
        temp_img = I_out[y:y+h,x:x+w]  
        p = percentile(temp_img)
        if p < 120:

            f = (120 + p) / ((2 * p) +1e-6)
            f = np.clip(f,1+1e-6,2-11e-6)
            B[y:y + h, x:x + w][skin_in_face > 0] = f
            B = eacp(B, lum)
            I_out = I_out2*B 

    final = I_out+detail
    final = np.stack((final, alpha, beta), axis=2)
    final = final.astype("uint8")
    final_bgr = cv.cvtColor(final, cv.COLOR_Lab2BGR)            
    cv.imshow('',final_bgr)
    cv.waitKey(0)           


def skyline(mask):
    h, w = mask.shape
    for i in range(w):
        raw = mask[:, i]
        after_median = medfilt(raw, 19)
        try:
            first_zero_index = np.where(after_median == 0)[0][0]
            first_one_index = np.where(after_median == 1)[0][0]
            if first_zero_index > 20:
                mask[first_one_index:first_zero_index, i] = 1
                mask[first_zero_index:, i] = 0
                mask[:first_one_index, i] = 0
        except:
            continue
    return mask

def check_blue(mask,a,b,c):
    b,g,r= np.array_split(mask, 3, axis=2)
    b = b[:, :, 0]
    g = g[:, :, 0]
    r = r[:, :, 0]
    IDEAL_SKY_BGR = (195, 165, 80)
    RANGE = (a,b,c)
    h,w = b.shape
    for i in range(h):
        for j in range(w):
            if not (abs(b[i][j]-IDEAL_SKY_BGR[0]) <RANGE[0] and abs(g[i][j]-IDEAL_SKY_BGR[1]) <RANGE[1] and abs(r[i][j]-IDEAL_SKY_BGR[2] )<RANGE[2]):
                b[i][j] =0
                g[i][j] =0
                r[i][j]=0
    return np.stack((b, g, r), axis=2)

def sky_mask(img,a=60,b=65,c=80):
    h, w, _ = img.shape

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_gray = cv.blur(img_gray, (9, 3))
    cv.medianBlur(img_gray, 5)
    lap = cv.Laplacian(img_gray, cv.CV_8U)
    gradient_mask = (lap < 5.9).astype(np.uint8)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))

    mask = cv.morphologyEx(gradient_mask, cv.MORPH_ERODE, kernel)
    # plt.imshow(mask)
    # plt.show()
    mask = skyline(mask)
    after_img = cv.bitwise_and(img, img, mask=mask)
    after_img = check_blue(after_img,a,b,c)
    return after_img


#def sky_correction

#def salient_correction

def final_image(img):
    skin_mask = detected_skin(img)
    sky_mask = sky_mask(img)
    img_1 = face_correction(img,skin_mask,0.5)
    #img_2 = sky_correction()
    #img_3 = salient_correction()



plt.show()


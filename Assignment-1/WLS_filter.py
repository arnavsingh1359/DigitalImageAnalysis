import cv2
import numpy
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt



def wlsfilter(image_orig, lambda_=0.4, alpha=1.2, small_eps=1e-4):
    """
    ARGs:
    -----
    image: 0-255, uint8, single channel (e.g. grayscale or single L)
    lambda_:
    alpha:
    RETURN:
    -----
    out: base, 0-1, float
    detail: detail, 0-1, float
    """
    # print 'wls: lambda, alpha', lambda_, alpha
    image = image_orig
    s = image.shape

    k = numpy.prod(s)

    dy = numpy.diff(image, 1, 0)
    dy = -lambda_ / (numpy.absolute(dy)**alpha + small_eps)
    last_y = numpy.zeros((s[1],))
    dy = numpy.vstack((dy, last_y))
    dy = dy.flatten()

    dx = numpy.diff(image, 1, 1)
    dx = -lambda_ / (numpy.absolute(dx) ** alpha + small_eps)
    dx = numpy.hstack((dx, numpy.zeros(s[0], )[:, numpy.newaxis]))
    dx = dx.flatten()

    a = spdiags(numpy.vstack((dx, dy)), [-s[0], -1], k, k)

    d = 1 - (dx + numpy.roll(dx, s[0]) + dy + numpy.roll(dy, 1))
    a = a + a.T + spdiags(d, 0, k, k )
    _out = spsolve(a, image.flatten()).reshape(s[::-1])
    out = numpy.rollaxis(_out,1)
    detail = image - out
    return out, detail 

def test_wlsfilter():
    """deprecated, need to remove rollaxis"""
    lambda_ = 0.1
    alpha = 1.2
    image = cv2.imread("Pictures/part1/img_6.png")
    # normalize image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out, detail = wlsfilter(image, lambda_, alpha)
    print(image.shape,out.shape,detail.shape)
    plt.imshow(detail,'gray')
    plt.show()

def test_luminance():
    lambda_ = 0.1
    alpha = 1.2
    image = cv2.imread('input_teaser.png')
    image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # don't forget to normalize
    image_lumi = image_LAB[...,0].astype(numpy.float)/255.0
    out, detail = wlsfilter(image_lumi, lambda_, alpha)
    image_base = numpy.zeros(image.shape)
    image_base[..., 0] = (numpy.rollaxis(out,1) *255.0)
    image_base[..., 1] = image_LAB[...,1]
    image_base[..., 2] = image_LAB[...,2]
    numpy.clip(image_base, 0, 255, out=image_base)
    image_base = image_base.astype('uint8')

    image_detail = numpy.zeros(image.shape)
    image_detail[..., 0] = (numpy.rollaxis(detail,1) *255.0)
    image_detail[..., 1] = image_LAB[...,1]
    image_detail[..., 2] = image_LAB[...,2]
    numpy.clip(image_detail, 0, 255, out=image_detail)
    image_detail = image_detail.astype('uint8')

    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_base_RGB = cv2.cvtColor(image_base, cv2.COLOR_LAB2RGB)
    image_detail_RGB = cv2.cvtColor(image_detail, cv2.COLOR_LAB2RGB)
    plt.imshow(numpy.hstack(
        [image_RGB, image_base_RGB, image_detail_RGB]
    ), cmap='jet')
    plt.show()

test_wlsfilter()

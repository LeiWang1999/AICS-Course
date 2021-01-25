import cv2 as cv

def crop_img(im, dist_shape):
    raw_shape = im.shape
    raw_height = raw_shape[0]
    raw_weight = raw_shape[1]
    dist_height = dist_shape[0]
    dist_weight = dist_shape[1]
    assert raw_height > dist_height and raw_weight > dist_weight , "input image shape must larger than dist"

    a = int(raw_height / 2 - dist_height / 2)
    b = int(raw_height / 2 + dist_height / 2)
    c = int(raw_weight / 2 - dist_weight / 2)
    d = int(raw_weight / 2 + dist_weight / 2)
    
    cropped_im = im[a:b, c:d]

    return cropped_im

def _processA(imagepath='hare.jpg'):
    im = cv.imread(imagepath)
    # 先resize到256，256
    im = cv.resize(im, (256, 256))
    im = crop_img(im, (224,224))
    cv.imshow('processA', im)
    cv.waitKey(0)


def _processB(imagepath='hare.jpg'):
    im = cv.imread(imagepath)
    im = crop_img(im, (int(im.shape[0] * 0.875), int(im.shape[1] * 0.875)))
    im = cv.resize(im, (224,224))
    cv.imshow('processB', im)
    cv.waitKey(0)

if __name__ == '__main__':
    _processA()
    _processB()
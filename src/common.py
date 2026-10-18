import numpy as np

def mask(img):
    #img has shape of(N,64,64)
    masked=img.copy()
    masked[:,24:40,24:40]=0
    return masked

def flatten(img):
    #img has a shape of (N,64,64)
    tempImg=img.copy()
    flat=tempImg.reshape(tempImg.shape[0], -1)
    return flat

def unflatten(img):
    #img has a shape of (N,4096)
    tempImg=img.copy()
    unflat=tempImg.reshape(-1,64,64)
    return unflat

if __name__ == "__main__":
    pass
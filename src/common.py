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

def loader():
    loaded=np.load("./data/dataset.npz")
    return loaded["training"], loaded["validation"], loaded["testing"]

def get_flat_images(index_array, data):
    #data has shape (400,64,64)
    return flatten(data[index_array])

def get_masked_images(index_array, data):
    #data has shape (400,64,64)
    return flatten(mask(data[index_array]))


if __name__ == "__main__":
    pass
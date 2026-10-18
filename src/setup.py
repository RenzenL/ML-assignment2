import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from common import flatten,mask

facesData = fetch_olivetti_faces()
print(facesData.keys())

seed_rng = np.random.default_rng(26007)
print(facesData['images'].shape)
#req list
#we store indexes in these
training=[]
validation=[]
testing=[]
for i in range(0,400,10) :
    order=seed_rng.permutation(10)
    for k in range(10):
        if k <6 :
            training.append(i+order[k])
        elif k<8 :
            validation.append(i+order[k])
        else : 
            testing.append(i+order[k])
#converting to numpy array
training=np.array(training)
validation=np.array(validation)
testing=np.array(testing)

assert(len(training)==240)
assert(len(validation)==80)
assert(len(testing)==80)
assert np.all(np.bincount(facesData.target[training])==6)
assert np.all(np.bincount(facesData.target[validation])==2)
assert np.all(np.bincount(facesData.target[testing])==2)
assert(len(np.unique(facesData.target[training]))==40)
assert(len(np.unique(facesData.target[testing]))==40)
assert(len(np.unique(facesData.target[validation]))==40)
joined=np.concatenate([testing,validation,training])
assert(np.array_equal(np.sort(joined),np.arange(400)))


np.savez("./data/dataset.npz", training=training, testing=testing, validation=validation)
loaded=np.load("./data/dataset.npz")
assert(np.array_equal(loaded["training"],training))
assert(np.array_equal(loaded["testing"],testing))
assert(np.array_equal(loaded["validation"],validation))

#masking and flattening the data
unflat_unmasked=facesData["images"][testing]
unflat_masked=mask(unflat_unmasked)
flat_masked=flatten(unflat_masked)
flat_unmasked=flatten(unflat_unmasked)
assert flat_masked.shape ==(80,4096)
assert np.array_equal(flat_unmasked, facesData["data"][testing])

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(unflat_unmasked[0], cmap="gray")    # a clean 2D image
axes[0].set_title("clean")
axes[1].imshow(unflat_masked[0], cmap="gray")   # the same image, masked
axes[1].set_title("masked")
for ax in axes:
    ax.axis("off")
plt.show()


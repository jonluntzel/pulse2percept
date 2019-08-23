import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from skimage.transform import rotate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# image is ndarray
img = io.imread("Landolt5.png")
x = img[200:1000, 200:1000].flatten()
# print(x)

# 2 samples per class: 2nd is a linear combination of the 1st and
# additional noise
y = np.linspace(-200, 200, 1920000)
y = y + x

# 1st row of training data
A = np.array([img[200:1000, 200:1000].flatten(), y])

theta = 90
for i in range(1, int(360 / theta)):
    new_img = rotate(img, theta * i)
    new_img = new_img[200:1000, 200:1000]
    x = new_img.flatten()
    y = np.linspace(-200, 200, 1920000)
    y = y + x
    A = np.vstack([A, [new_img.flatten(), y]])
# print(A)

x = []
for i in range(0, int(360 / theta)):
    x.append(i * theta)
    x.append(i * theta)
print(x)

clf = LinearDiscriminantAnalysis()
clf.fit(A, x)

z = img[200:1000, 200:1000].flatten()
print(clf.predict([z, z]))
z = rotate(img, 90)
z = z[200:1000, 200:1000].flatten()
print(clf.predict([z, z]))
z = rotate(img, 180)
z = z[200:1000, 200:1000].flatten()
print(clf.predict([z, z]))
z = rotate(img, 270)
z = z[200:1000, 200:1000].flatten()
print(clf.predict([z, z]))

# 1 sample per class?
# img = io.imread("Landolt5.png")
# x = img[200:1000, 200:1000].flatten()
# A = np.array(img[200:1000, 200:1000].flatten())
# # img = img[200:1000, 200:1000]
# for i in range(1, 8):
#     new_img = rotate(img, 45 * i)
#     new_img = new_img[200:1000, 200:1000]
#     x = new_img.flatten()
#     A = np.vstack([A, new_img.flatten()])
# # [1, 2, 3, 4, 5, 6, 7, 8]
# b = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8])
# clf = LinearDiscriminantAnalysis()
# clf.fit(A, b)


# looking at rings
# print(img.shape)
# img = img[0:1360, 0:1360]
# img = resize(img, (16, 16))
#new_img = rotate(img, 45)
#new_img = new_img[200:1000, 200:1000]
# io.imshow(img)
# plt.show()
# io.imshow(new_img)
# plt.show()

#img2 = rotate(img, 135)
#img2 = img2[200:1000, 200:1000]
# io.imshow(img2)
# plt.show()
# print(img2.flatten())
# new_img = rotate(img, 135)
# new_img = new_img[362:968, 385:988]
# io.imshow(new_img)
# plt.show()
# numpy.array(img.flatten() for img in (list of images))

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.transform import resize
from skimage.transform import rotate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# i think i'll do scaling last
# image is ndarray
sigma = 0.045  # .45  # 1  # 0.00043
img = io.imread("Landolt5.png")
x = img[200:1000, 200:1000].flatten()
x2 = img[300:1000, 200:1000].flatten()
# io.imshow(img[300:1000, 200:1000])
# plt.show()
# io.imshow(rotate(img, 180)[280:1000, 200:1000])
# plt.show()
# print(x)

# collections of variations on the original image
# img2 = resize(img, (16, 16))
# io.imshow(img2)
# plt.show()
# img2 = resize(img, (32, 32))
# io.imshow(img2)
# plt.show()
# img2 = resize(img, (64, 64))
# io.imshow(img2)
# plt.show()

# translations (works for all orientations)
# t up
# img2 = img[280:1080, 200:1000]
# t down
# img2 = img[120:920, 200:1000]  # 150:950 for no black corners
# t right
# io.imshow(rotate(img, 135)[200:1000, 150:950])
# t left
# io.imshow(rotate(img, 45)[200:1000, 250:1050])
# plt.show()

# training data: resize and translate, combine different things.

# to translate and compress, simply feed in the dimensions determined above into the compressor,
# and a) see what that does b) feed in the cropped image with normal abcd vaues


def compressor(img, factor, a, b, c, d):
    im2 = img.copy()
    im2slice = im2[a:b, c:d]

    im2slice = im2slice.compress([not(i % factor)
                                  for i in range(len(im2slice))], axis=0)
    im2slice = im2slice.compress([not(i % factor)
                                  for i in range(len(im2slice[0]))], axis=1)

    im2[a:b, c:d] = 255

    width = (b - a) // factor
    height = (d - c) // factor
    row_midpoint = (a + b) // 2
    col_midpoint = (c + d) // 2

    im2[row_midpoint - width // 2:row_midpoint + width // 2,
        col_midpoint - height // 2:col_midpoint + height // 2] = im2slice

    return im2

# 2 samples per class: 2nd is a linear combination of the 1st and
# additional noise
y = np.random.randn(48, 1920000) * sigma  # (16, 1920000) * sigma
# y = y + x

# 1st row of training data
# A = np.array([x, x, x, x])  # np.array([img[200:1000,
# 200:1000].flatten(), y])
A = np.array([x, x, x, x, x, x, x, x, x, x, x, x])
print(A)
print(x.max())


# img2 = img[280:1080, 200:1000]
# t down
# img2 = img[120:920, 200:1000]  # 150:950 for no black corners
# t right
# io.imshow(rotate(img, 135)[200:1000, 150:950])
# t left
# io.imshow(rotate(img, 45)[200:1000, 250:1050])
# plt.show()

theta = 90
for i in range(1, int(360 / theta)):
    new_img = rotate(img, theta * i)
    rot = new_img[200:1000, 200:1000].flatten()
    # x = rot_img.flatten()

    # new additions
    # translations in each direction
    up = new_img[280:1080, 200:1000].flatten()
    down = new_img[150:950, 200:1000].flatten()
    right = new_img[200:1000, 150:950].flatten()
    left = new_img[200:1000, 250:1050].flatten()

    # compressions: 2x and 4x
    compressed = compressor(img, 2, 200, 1000, 200, 1000)[
        200:1000, 200:1000].flatten()
    # y = np.random.randn(1920000) * sigma
    # y = np.linspace(-200, 200, 1920000)  # random.randn
    # y = y + x
    # np.vstack([A, [new_img.flatten(), y]])

    # A = np.vstack([A, [x, x, x, x]])
    # 2 repeats?
    A = np.vstack([A, [rot, rot, up, up, down, down, right,
                       right, left, left, compressed, compressed]])  # left, left, left,
    # left]])  # compressed, compressed]])
# print(A)

x = []
for i in range(0, int(360 / theta)):
    for j in range(0, 12):  # range(0, 4):
        x.append(i * theta)
print(x)

A = A + y
clf = LinearDiscriminantAnalysis()
clf.fit(A, x)

z = img[200: 1000, 200: 1000].flatten()
print(clf.predict([z, z]))
z = rotate(img, 90)
z = z[200: 1000, 200: 1000].flatten()
print(clf.predict([z, z]))
z = rotate(img, 180)
z = z[200: 1000, 200: 1000].flatten()
print(clf.predict([z, z]))
z = rotate(img, 270)
z = z[200: 1000, 200: 1000].flatten()
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
# new_img = rotate(img, 45)
# new_img = new_img[200:1000, 200:1000]
# io.imshow(img)
# plt.show()
# io.imshow(new_img)
# plt.show()

# img2 = rotate(img, 135)
# img2 = img2[200:1000, 200:1000]
# io.imshow(img2)
# plt.show()
# print(img2.flatten())
# new_img = rotate(img, 135)
# new_img = new_img[362:968, 385:988]
# io.imshow(new_img)
# plt.show()
# numpy.array(img.flatten() for img in (list of images))

# def scale_phosphene(img, scale):
#     m = skim.moments(img, order=1)
#     # Shift the phosphene to (0, 0):
#     transl = np.array([-m[1, 0] / m[0, 0], -m[0, 1] / m[0, 0]])
#     tf_shift = skit.SimilarityTransform(translation=transl)
#     # Scale the phosphene:
#     tf_scale = skit.SimilarityTransform(scale=scale)
#     # Shift the phosphene back to where it was:
#     tf_shift_inv = skit.SimilarityTransform(translation=-transl)
#   return skit.warp(img, (tf_shift + (tf_scale + tf_shift_inv)).inverse)

# g = scale_phosphene(img, 1)

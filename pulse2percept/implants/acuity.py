import numpy as np
import matplotlib.pyplot as plt

from skimage import color
from skimage import io
from skimage.transform import resize
from skimage.transform import rotate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# noise function testing
import os
import cv2

# random
import random
# load method
from PIL import Image
import pickle

# test set: do translations that are random values between 0 and strong shift. include 8x compressions.
# image is ndarray
# low sigma related to pixel values ranging from 0 to 255?
# class StimulusLDA():
sigma = 0  # .0045  # .45  # 1  # 0.00043
theta = 90
img = io.imread("Landolt5.png")
print(img.shape)

x = img[200:1000, 200:1000].flatten()

#, cmap='gray')
plt.title('compresssss')
# [200:1000, 200:1000].flatten()
# c = compressor(C, 200, 1000, 200, 1000)
# just temporary tests
# samples = np.array([x])
samples = np.vstack([[x], [x]])
print(samples.shape)

a = np.array([[x], [x], [x]])
b = np.array([[x], [x], [x]])
c = np.vstack((a, b))
print(c.shape)
print(np.vstack([[x], [x], [x]]))

# method by Shubham Pachori, stackoverflow
# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv


# def noisy(noise_typ, image):
#     if noise_typ == "gauss":
#         row, col, ch = image.shape
#         mean = 0
#         var = 0.1
#         sigma = var**0.5
#         gauss = np.random.normal(mean, sigma, (row, col, ch))
#         gauss = gauss.reshape(row, col, ch)
#         noisy = image - gauss
#         return noisy
#     elif noise_typ == "s&p":
#         row, col, ch = image.shape
#         s_vs_p = 0.5
#         amount = 0.004
#         out = np.copy(image)
#         # Salt mode
#         num_salt = np.ceil(amount * image.size * s_vs_p)
#         coords = [np.random.randint(0, i - 1, int(num_salt))
#                   for i in image.shape]
#         out[coords] = 1

#         # Pepper mode
#         num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
#         coords = [np.random.randint(0, i - 1, int(num_pepper))
#                   for i in image.shape]
#         out[coords] = 0
#         return out
#     elif noise_typ == "poisson":
#         vals = len(np.unique(image))
#         vals = 2 ** np.ceil(np.log2(vals))
#         noisy = np.random.poisson(image * vals) / float(vals)
#         return noisy
#     elif noise_typ == "speckle":
#         row, col, ch = image.shape
#         gauss = np.random.randn(row, col, ch)
#         gauss = gauss.reshape(row, col, ch)
#         noisy = image + image * gauss
#         return noisy

# img2 = noisy('gauss', img)
# io.imshow(img2)
# plt.show()

# img2 = noisy('speckle', img)
# io.imshow(img2)
# plt.show()

# img2 = color.rgb2gray(io.imread("Landolt5.png")[200:1000, 200:1000])
# #y = np.random.rand(800, 800) * sigma - sigma / 2
# img2 = np.minimum(1, np.maximum(0, img2 + y))
# io.imshow(img2)
# plt.show()
# for i in range(0, 800):
#     for j in range(0, 800):
#         x = random.randint(0, 256)
#         if (img2[i, j] == 0):
#             img2[i, j] = img2[i, j] + x
#         else:
#             img2[i, j] = img2[i, j] - x

# io.imshow(img2)
# plt.show()

# img2 = noisy('gauss', noisy('gauss', noisy('gauss', noisy('gauss', img))))
# img2 = noisy('speckle', noisy('gauss', img))
# img2 = noisy('')
# io.imshow(img2)
# plt.show()
# img2 = noisy('speckle', noisy(
#     'speckle', noisy('speckle', noisy('speckle', img))))
# io.imshow(img2)
# plt.show()


def compressor(img, a, b, c, d):
    im2 = img.copy()
    im2slice = im2[a:b, c:d]

    im2slice = im2slice.compress([not(i % 2)
                                  for i in range(len(im2slice))], axis=0)
    im2slice = im2slice.compress([not(i % 2)
                                  for i in range(len(im2slice[0]))], axis=1)

    im2[a:b, c:d] = 255

    width = (b - a) // 2
    height = (d - c) // 2
    row_midpoint = (a + b) // 2
    col_midpoint = (c + d) // 2

    im2[row_midpoint - width // 2:row_midpoint + width // 2,
        col_midpoint - height // 2:col_midpoint + height // 2] = im2slice

    return im2

# down
# test = np.vstack([img[0:100, :], img])
# io.imshow(test)
# plt.show()
# # up
# test = np.vstack([img, img[0:100, :]])
# io.imshow(test)
# plt.show()
# # right
# test = np.hstack([img[:, 0:100], img])
# io.imshow(test)
# plt.show()
# # left
# test = np.hstack([img, img[:, 0:100]])
# io.imshow(test)
# plt.show()

# fenceposting to see if it mitigatess the "setting an array element with
# a sequence"

# game plan: look for dimensions that are not 800 x 800, and review in the
# notebook how to correct it
# once LDA works, comment out the fencepost, add conditionals, and check
# new_img = img
# rot = new_img[200:1000, 200:1000]  # .flatten()

# U = new_img[250:1050, 200:1000]  # .flatten()
# D = new_img[150:950, 200:1000]  # .flatten()
# R = new_img[200:1000, 150:950]  # .flatten()
# L = new_img[200:1000, 250:1050]  # .flatten()

# u = new_img[225:1025, 200:1000]  # .flatten()
# d = new_img[175:975, 200:1000]  # .flatten()
# r = new_img[200:1000, 175:975]  # .flatten()
# l = new_img[200:1000, 225:1025]  # .flatten()

# C = compressor(new_img, 200, 1000, 200, 1000)
# # [200:1000, 200:1000].flatten()
# c = compressor(C, 200, 1000, 200, 1000)
# # C = C[200:1000, 200:1000].flatten()

# # witchcraft necessary for strong shifts at 4x compression. I need to
# # think deliberately through these. make the base images large, and cut
# # from them.
# up = np.vstack([c, img[0:100, :]])
# down = np.vstack([c[0:100, :], c])
# right = np.hstack([c[:, 0:100], c[:, 0:700]])
# left = np.hstack([c, img[:, 0:100]])

# upright = np.hstack([up[:, 0:100], up])
# upleft = np.hstack([up, up[:, 0:100]])
# downright = np.hstack([c[0:800, 0:100], down[0:800, 0:700]])
# downleft = np.hstack([down, up[:, 0:100]])

# # UC, uC, Uc, uc, ... D.L.R (16)
# U_C = C[400:1200, 200:1000]  # .flatten()
# uC = C[300:1100, 200:1000]  # .flatten()
# # Uc = c[500:1300, 200:1000].flatten() #***
# Uc = up[500:1300, 200:1000]  # .flatten()
# uc = c[350:1150, 200:1000]  # .flatten()

# D_C = C[0:800, 200:1000]  # .flatten()
# dC = C[100:900, 200:1000]  # .flatten()
# Dc = down[0:800, 200:1000]  # .flatten()
# dc = c[50:850, 200:1000]  # .flatten()

# R_C = C[200:1000, 0:800]  # .flatten()
# rC = C[200:1000, 100:900]  # .flatten()
# Rc = right[200:1000, :]  # .flatten()
# # Rc = c[250:950, 0:700].flatten()
# rc = c[200:1000, 50:850]  # .flatten()

# L_C = C[200:1000, 400:1200]  # .flatten()
# lC = C[200:1000, 300:1100]  # .flatten()
# # Lc = c[200:1000, 500:1300].flatten()  # ***
# Lc = left[200:1000, 500:1300]  # .flatten()
# lc = c[200:1000, 350:1150]  # .flatten()

# # UL, uL, ul, Ul, UR, uR, ur, Ur, DL, dL, dl, Dl, DR, dR, dr, Dr (16)
# U_L = new_img[250:1050, 250:1050]  # .flatten()
# uL = new_img[225:1025, 250:1050]  # .flatten()
# ul = new_img[225:1025, 225:1025]  # .flatten()
# Ul = new_img[250:1050, 225:1025]  # .flatten()

# U_R = new_img[250:1050, 150:950]  # .flatten()
# uR = new_img[225:1025, 150:950]  # .flatten()
# ur = new_img[225:1025, 175:975]  # .flatten()
# Ur = new_img[250:1050, 175:975]  # .flatten()

# D_L = new_img[150:950, 250:1050]  # .flatten()
# dL = new_img[175:975, 250:1050]  # .flatten()
# dl = new_img[175:975, 225:1025]  # .flatten()
# Dl = new_img[150:950, 225:1025]  # .flatten()

# D_R = new_img[150:950, 150:950]  # .flatten()
# dR = new_img[175:975, 150:950]  # .flatten()
# dr = new_img[175:975, 175:975]  # .flatten()
# Dr = new_img[150:950, 175:975]  # .flatten()

# # (UL uL ul Ul)(C, c), (UR, uR, ur, Ur)(C,c), ... (32)
# # just named, none of the slices are accurate yet
# U_L_C = C[400:1200, 400:1200]  # .flatten()
# uLC = C[300:1100, 400:1200]  # .flatten()
# ulC = C[300:1100, 300:1100]  # .flatten()
# UlC = C[400:1200, 300:1100]  # .flatten()

# U_R_C = C[400:1200, 0:800]  # .flatten()
# uRC = C[300:1100, 0:800]  # .flatten()
# urC = C[300:1100, 100:900]  # .flatten()
# UrC = C[400:1200, 100:900]  # .flatten()

# D_L_C = C[0:800, 400:1200]  # .flatten()
# dLC = C[100:900, 400:1200]  # .flatten()
# dlC = C[100:900, 300:1100]  # .flatten()
# DlC = C[0:800, 300:1100]  # .flatten()

# D_R_C = C[0:800, 0:800]  # .flatten()
# dRC = C[100:900, 0:800]  # .flatten()
# drC = C[100:900, 100:900]  # .flatten()
# DrC = C[0:800, 100:900]  # .flatten()

# # 2 shifts, 4x
# # ULc = c[500:1300, 500:1300].flatten()  # ***
# ULc = upleft[500:1300, 500:1300]  # .flatten()
# # uLc = c[350:1150, 500:1300].flatten()  # ***
# uLc = left[350:1150, 500:1300]  # .flatten()
# ulc = c[350:1150, 350:1150]  # .flatten()
# # Ulc = c[500:1300, 350:1150].flatten()  # ***
# Ulc = up[500:1300, 350:1150]  # .flatten()

# # URc = right[500:1300, :].flatten()  # *** np.hstack([img, img[:, 0:100]])
# URc = upright[500:1300, 0:800]  # .flatten()
# uRc = right[350:1150, :]  # .flatten()
# urc = c[350:1150, 50:850]  # .flatten()
# # Urc = c[500:1300, 50:850].flatten()  # ***
# Urc = up[500:1300, 50:850]  # .flatten()

# # DLc = down[:, 500:1300].flatten() #***
# # np.hstack([down, img[:, 0:100]])[200:1000, 500:1300].flatten()
# DLc = downleft[0:800, 500:1300]  # .flatten()
# dLc = left[50:850, 500:1300]  # .flatten()  # ***
# dlc = c[50:850, 350:1150]  # .flatten()
# Dlc = down[0:800, 350:1150]  # .flatten()

# DRc = downright  # .flatten()
# dRc = right[50:850, :]  # .flatten()
# drc = c[50:850, 50:850]  # .flatten()
# Drc = down[0:800, 50:850]  # .flatten()

# C = C[200:1000, 200:1000]  # .flatten()
# c = c[200:1000, 200:1000]  # .flatten()

# # 13, 25
# # 17, 18, 20, 21, 24, 25, 26,
# # ones to fix. add white.
# # print(Dc.shape, Drc.shape, Dlc.shape)
# # print(Uc.shape, Lc.shape, ULc.shape, uLc.shape, Ulc.shape,
# #       URc.shape, Urc.shape, DLc.shape, dLc.shape)

# # print(rot.shape, U.shape, D.shape, R.shape, L.shape, u.shape, d.shape, r.shape, l.shape, C.shape, c.shape, U_C.shape, uC.shape, Uc.shape, uc.shape, D_C.shape, dC.shape, Dc.shape, dc.shape, R_C.shape, rC.shape, Rc.shape, rc.shape, L_C.shape, lC.shape, Lc.shape, lc.shape, U_L.shape, uL.shape, ul.shape, Ul.shape, U_R.shape, uR.shape, ur.shape, Ur.shape, D_L.shape, dL.shape, dl.shape, Dl.shape,
# # D_R.shape, dR.shape, dr.shape, Dr.shape, U_L_C.shape, uLC.shape,
# # ulC.shape, UlC.shape, U_R_C.shape, uRC.shape, urC.shape, UrC.shape,
# # D_L_C.shape, dLC.shape, dlC.shape, DlC.shape, D_R_C.shape, dRC.shape,
# # drC.shape, DrC.shape, ULc.shape, uLc.shape, ulc.shape, Ulc.shape,
# # URc.shape, uRc.shape, urc.shape, Urc.shape, DLc.shape, dLc.shape,
# # dlc.shape, Dlc.shape, DRc.shape, dRc.shape, drc.shape, Drc.shape)

# A = np.vstack([[rot], [U], [D], [R], [L], [u], [d], [r], [l], [C], [c], [U_C], [uC], [Uc], [uc], [D_C], [dC], [Dc], [dc], [R_C], [rC], [Rc], [rc], [L_C], [lC], [Lc], [lc], [U_L], [uL], [ul], [Ul], [U_R], [uR], [ur], [Ur], [D_L], [dL], [dl], [Dl], [D_R], [dR], [dr],
#                [Dr], [U_L_C], [uLC], [ulC], [UlC], [U_R_C], [uRC], [urC], [UrC], [D_L_C], [dLC], [dlC], [DlC], [D_R_C], [dRC], [drC], [DrC], [ULc], [uLc], [ulc], [Ulc], [URc], [uRc], [urc], [Urc], [DLc], [dLc], [dlc], [Dlc], [DRc], [dRc], [drc], [Drc]])
# print(A.shape)
# A = np.array([rot, U, D, R, L, u, d, r, l, C, c, U_C, uC, Uc, uc, D_C, dC, Dc, dc, R_C, rC, Rc, rc, L_C, lC, Lc, lc, U_L, uL, ul, Ul, U_R, uR, ur, Ur, D_L, dL, dl, Dl, D_R, dR, dr, Dr, U_L_C,
# uLC, ulC, UlC, U_R_C, uRC, urC, UrC, D_L_C, dLC, dlC, DlC, D_R_C, dRC,
# drC, DrC, ULc, uLc, ulc, Ulc, URc, uRc, urc, Urc, DLc, dLc, dlc, Dlc,
# DRc, dRc, drc, Drc])  # compressed,


# def buildStimuli(img, theta):
# UDLR = strong translation (up, down, left, right)
# udlr = weak translation
# strong: for 1:1 image, 50 pixels. 2:1, 200 pixels. 4:1, ...
# weak: (strong # pixels) / 2
# C = 2x Compression vertically & horizontally, c = 4x

# 75 per row
# for each rotation:
# rot (normal), U, D ,L, R, u, d, l, r, C, c (11)
# UC, uC, Uc, uc, D.L.R (16)
# UL uL ul Ul, UR, uR, ur, Ur, DL, dL, dl, Dl, DR, dR, dr, Dr (16)
# (UL uL ul Ul)(C, c), (UR, uR, ur, Ur)(C,c), ... (32)

# once you've finished, double check it in jupyter notebook
# samples = []
for i in range(0, 1):  # int(360 / 90)):  # 1 = 0, 90 = theta
    new_img = rotate(img, 90 * i)  # 90 = theta
    rot = new_img[200:1000, 200:1000].flatten()

    U = new_img[250:1050, 200:1000].flatten()
    D = new_img[150:950, 200:1000].flatten()
    R = new_img[200:1000, 150:950].flatten()
    L = new_img[200:1000, 250:1050].flatten()

    u = new_img[225:1025, 200:1000].flatten()
    d = new_img[175:975, 200:1000].flatten()
    r = new_img[200:1000, 175:975].flatten()
    l = new_img[200:1000, 225:1025].flatten()

    C = compressor(new_img, 200, 1000, 200, 1000)
    # [200:1000, 200:1000].flatten()
    c = compressor(C, 200, 1000, 200, 1000)
    # C = C[200:1000, 200:1000].flatten()

    # witchcraft necessary for strong shifts at 4x compression. I need to
    # think deliberately through these. make the base images large, and cut
    # from them.
    up = np.vstack([c, img[0:100, :]])
    down = np.vstack([c[0:100, :], c])
    right = np.hstack([c[:, 0:100], c[:, 0:700]])
    left = np.hstack([c, img[:, 0:100]])

    upright = np.hstack([up[:, 0:100], up])
    upleft = np.hstack([up, up[:, 0:100]])
    downright = np.hstack([c[0:800, 0:100], down[0:800, 0:700]])
    downleft = np.hstack([down, up[:, 0:100]])

    # UC, uC, Uc, uc, ... D.L.R (16)
    U_C = C[400:1200, 200:1000].flatten()
    uC = C[300:1100, 200:1000].flatten()
    # Uc = c[500:1300, 200:1000].flatten() #***
    Uc = up[500:1300, 200:1000].flatten()
    uc = c[350:1150, 200:1000].flatten()

    D_C = C[0:800, 200:1000].flatten()
    dC = C[100:900, 200:1000].flatten()
    Dc = down[0:800, 200:1000].flatten()
    dc = c[50:850, 200:1000].flatten()

    R_C = C[200:1000, 0:800].flatten()
    rC = C[200:1000, 100:900].flatten()
    Rc = right[200:1000, :].flatten()
    # Rc = c[250:950, 0:700].flatten()
    rc = c[200:1000, 50:850].flatten()

    L_C = C[200:1000, 400:1200].flatten()
    lC = C[200:1000, 300:1100].flatten()
    # Lc = c[200:1000, 500:1300].flatten()  # ***
    Lc = left[200:1000, 500:1300].flatten()
    lc = c[200:1000, 350:1150].flatten()

    # UL, uL, ul, Ul, UR, uR, ur, Ur, DL, dL, dl, Dl, DR, dR, dr, Dr (16)
    U_L = new_img[250:1050, 250:1050].flatten()
    uL = new_img[225:1025, 250:1050].flatten()
    ul = new_img[225:1025, 225:1025].flatten()
    Ul = new_img[250:1050, 225:1025].flatten()

    U_R = new_img[250:1050, 150:950].flatten()
    uR = new_img[225:1025, 150:950].flatten()
    ur = new_img[225:1025, 175:975].flatten()
    Ur = new_img[250:1050, 175:975].flatten()

    D_L = new_img[150:950, 250:1050].flatten()
    dL = new_img[175:975, 250:1050].flatten()
    dl = new_img[175:975, 225:1025].flatten()
    Dl = new_img[150:950, 225:1025].flatten()

    D_R = new_img[150:950, 150:950].flatten()
    dR = new_img[175:975, 150:950].flatten()
    dr = new_img[175:975, 175:975].flatten()
    Dr = new_img[150:950, 175:975].flatten()

    # (UL uL ul Ul)(C, c), (UR, uR, ur, Ur)(C,c), ... (32)
    # just named, none of the slices are accurate yet
    U_L_C = C[400:1200, 400:1200].flatten()
    uLC = C[300:1100, 400:1200].flatten()
    ulC = C[300:1100, 300:1100].flatten()
    UlC = C[400:1200, 300:1100].flatten()

    U_R_C = C[400:1200, 0:800].flatten()
    uRC = C[300:1100, 0:800].flatten()
    urC = C[300:1100, 100:900].flatten()
    UrC = C[400:1200, 100:900].flatten()

    D_L_C = C[0:800, 400:1200].flatten()
    dLC = C[100:900, 400:1200].flatten()
    dlC = C[100:900, 300:1100].flatten()
    DlC = C[0:800, 300:1100].flatten()

    D_R_C = C[0:800, 0:800].flatten()
    dRC = C[100:900, 0:800].flatten()
    drC = C[100:900, 100:900].flatten()
    DrC = C[0:800, 100:900].flatten()

    # 2 shifts, 4x
    # ULc = c[500:1300, 500:1300].flatten()  # ***
    ULc = upleft[500:1300, 500:1300].flatten()
    # uLc = c[350:1150, 500:1300].flatten()  # ***
    uLc = left[350:1150, 500:1300].flatten()
    ulc = c[350:1150, 350:1150].flatten()
    # Ulc = c[500:1300, 350:1150].flatten()  # ***
    Ulc = up[500:1300, 350:1150].flatten()

    # URc = right[500:1300, :].flatten()  # *** np.hstack([img, img[:, 0:100]])
    URc = upright[500:1300, 0:800].flatten()
    uRc = right[350:1150, :].flatten()
    urc = c[350:1150, 50:850].flatten()
    # Urc = c[500:1300, 50:850].flatten()  # ***
    Urc = up[500:1300, 50:850].flatten()

    # DLc = down[:, 500:1300].flatten() #***
    # np.hstack([down, img[:, 0:100]])[200:1000, 500:1300].flatten()
    DLc = downleft[0:800, 500:1300].flatten()
    dLc = left[50:850, 500:1300].flatten()  # ***
    dlc = c[50:850, 350:1150].flatten()
    Dlc = down[0:800, 350:1150].flatten()

    DRc = downright.flatten()
    dRc = right[50:850, :].flatten()
    drc = c[50:850, 50:850].flatten()
    Drc = down[0:800, 50:850].flatten()

    C = C[200:1000, 200:1000].flatten()
    c = c[200:1000, 200:1000].flatten()

    if (i == 0):
        A = np.vstack([[rot], [U], [D], [R], [L], [u], [d], [r], [l], [C], [c], [U_C], [uC], [Uc], [uc], [D_C], [dC], [Dc], [dc], [R_C], [rC], [Rc], [rc], [L_C], [lC], [Lc], [lc], [U_L], [uL], [ul], [Ul], [U_R], [uR], [ur], [Ur], [D_L], [dL], [dl], [Dl], [D_R], [dR], [dr],
                       [Dr], [U_L_C], [uLC], [ulC], [UlC], [U_R_C], [uRC], [urC], [UrC], [D_L_C], [dLC], [dlC], [DlC], [D_R_C], [dRC], [drC], [DrC], [ULc], [uLc], [ulc], [Ulc], [URc], [uRc], [urc], [Urc], [DLc], [dLc], [dlc], [Dlc], [DRc], [dRc], [drc], [Drc]])
        # A = np.array([rot, U, D, R, L, u, d, r, l, C, c, U_C, uC, Uc, uc, D_C, dC, Dc, dc, R_C, rC, Rc, rc, L_C, lC, Lc, lc, U_L, uL, ul, Ul, U_R, uR, ur, Ur, D_L, dL, dl, Dl, D_R, dR, dr, Dr, U_L_C,
        # uLC, ulC, UlC, U_R_C, uRC, urC, UrC, D_L_C, dLC, dlC, DlC, D_R_C, dRC,
        # drC, DrC, ULc, uLc, ulc, Ulc, URc, uRc, urc, Urc, DLc, dLc, dlc, Dlc,
        # DRc, dRc, drc, Drc])  # compressed, compressed
    else:
        A = np.vstack([A, [rot], [U], [D], [R], [L], [u], [d], [r], [l], [C], [c], [U_C], [uC], [Uc], [uc], [D_C], [dC], [Dc], [dc], [R_C], [rC], [Rc], [rc], [L_C], [lC], [Lc], [lc], [U_L], [uL], [ul], [Ul], [U_R], [uR], [ur], [Ur], [D_L], [dL], [dl], [Dl], [D_R], [dR], [dr],
                       [Dr], [U_L_C], [uLC], [ulC], [UlC], [U_R_C], [uRC], [urC], [UrC], [D_L_C], [dLC], [dlC], [DlC], [D_R_C], [dRC], [drC], [DrC], [ULc], [uLc], [ulc], [Ulc], [URc], [uRc], [urc], [Urc], [DLc], [dLc], [dlc], [Dlc], [DRc], [dRc], [drc], [Drc]])
# A = np.vstack([A, [rot, U, D, R, L, u, d, r, l, C, c, U_C, uC, Uc, uc, D_C, dC, Dc, dc, R_C, rC, Rc, rc, L_C, lC, Lc, lc, U_L, uL, ul, Ul, U_R, uR, ur, Ur, D_L, dL, dl, Dl, D_R, dR, dr, Dr,
# U_L_C, uLC, ulC, UlC, U_R_C, uRC, urC, UrC, D_L_C, dLC, dlC, DlC, D_R_C,
# dRC, drC, DrC, ULc, uLc, ulc, Ulc, URc, uRc, urc, Urc, DLc, dLc, dlc,
# Dlc, DRc, dRc, drc, Drc]])  # left, left, left,
print(A.shape)

# sub = np.array([[rot], [U], [D], [R], [L], [u], [d], [r], [l], [C], [c], [U_C], [uC], [Uc], [uc], [D_C], [dC], [Dc], [dc], [R_C], [rC], [Rc], [rc], [L_C], [lC], [Lc], [lc], [U_L], [uL], [ul], [Ul], [U_R], [uR], [Ur], [ur], [D_L], [dL], [dl], [Dl], [D_R], [dR], [dr],
#                 [Dr], [U_L_C], [uLC], [ulC], [UlC], [U_R_C], [uRC], [urC], [UrC], [D_L_C], [dLC], [dlC], [DlC], [D_R_C], [dRC], [drC], [DrC], [ULc], [uLc], [ulc], [Ulc], [URc], [uRc], [urc], [Urc], [DLc], [dLc], [dlc], [Dlc], [DRc], [dRc], [drc], [Drc]])

# A = np.vstack((A, sub))  # left, left, left,

# return A

# A = buildStimuli(img, 90)
#pickle.dump(A, open("save.p", "wb"))
# A = pickle.load(open("save.p", "rb"))
print(A.shape)


# compressed = compressor(img, 200, 1000, 200, 1000)[
#     200:1000, 200:1000]
# io.imshow(compressed)
# plt.show()

# compressed = compressor(compressor(img, 200, 1000, 200, 1000), 200, 1000, 200, 1000)[
#     200:1000, 200:1000]
# io.imshow(compressed)
# plt.show()


# 2 samples per class: 2nd is a linear combination of the 1st and
# additional noise: 48 total images, 800x800 values for each image to add
# noise to

# y = np.random.rand(800, 800) * 2 - 1
# y = np.random.randn(48, 1920000) * sigma - sigma / 2
# y = np.random.randn(48, 1920000) * sigma  # (16, 1920000) * sigma

# samples = []
# true_orient = []
# for rot in range():
#     for transl in range():
#         myimg = rotate(img, th)
#         samples.append(img.flatten())

# A = np.array(samples)
# print(A.shape)

# # manually building first row of matrix
# up = img[280:1080, 200:1000].flatten()
# down = img[150:950, 200:1000].flatten()
# right = img[200:1000, 150:950].flatten()
# left = img[200:1000, 250:1050].flatten()

# compressed = compressor(img, 200, 1000, 200, 1000)[
#     200:1000, 200:1000].flatten()

# A = np.array([x, x, up, up, down, down, right, right,
# left, left, compressed, compressed])  # , compressed, compressed])

# print(A)
# print(x.max())

# img2 = img[280:1080, 200:1000]
# t down
# img2 = img[120:920, 200:1000]  # 150:950 for no black corners
# t right
# io.imshow(rotate(img, 135)[200:1000, 150:950])
# t left
# io.imshow(rotate(img, 45)[200:1000, 250:1050])
# plt.show()

# samples = []
# for i in range(1, int(360 / theta)):  # 1 = 0
#     new_img = rotate(img, theta * i)
#     rot = new_img[200:1000, 200:1000].flatten()
#     # rot = noisy('gauss', noisy('speckle', new_img[
#     #            200:1000, 200:1000])).flatten()
#     # x = rot_img.flatten()

#     # new additions
#     # translations in each direction
#     up = new_img[280:1080, 200:1000].flatten()
#     down = new_img[150:950, 200:1000].flatten()
#     right = new_img[200:1000, 150:950].flatten()
#     left = new_img[200:1000, 250:1050].flatten()

#     # compressions: 2x and 4x
#     # for some reason I get a memory error when passing in new_img. I need to
#     # though
#     # ok now it's just terrible at classifying

#     compressed = compressor(new_img, 200, 1000, 200, 1000)[
#         200:1000, 200:1000]  # .flatten()
#     # io.imshow(compressed)
#     # plt.show()
#     compressed = compressed.flatten()
#     # ompressed = noisy('gauss', noisy('speckle', compressor(new_img, 2, 200, 1000, 200, 1000)[
#     #    200:1000, 200:1000])).flatten()
#     # y = np.random.randn(1920000) * sigma
#     # y = np.linspace(-200, 200, 1920000)  # random.randn
#     # y = y + x
#     # np.vstack([A, [new_img.flatten(), y]])

#     # A = np.vstack([A, [x, x, x, x]])
#     # 2 repeats?
#     # A = np.vstack([A, [rot, rot, up, up, down, down, right,
#     # right, left, left, compressed, compressed]])  # left, left, left,
#     # if (i == 0):
#     #    A = np.array([rot, rot, up, up, down, down, right,
#     #                  right, left, left, compressed, compressed])  # compressed, compressed
#     # print(A)
#     # else:
#     A = np.vstack([A, [rot, rot, up, up, down, down, right,
# right, left, left, compressed, compressed]])  # left, left, left,

#     # samples.extend((rot, rot, up, up, down, down, right,
#     #                 right, left, left, compressed, compressed))
#     # left]])  # compressed, compressed]])
# # print(samples)
# # # A = np.vstack(samples)
# # print(A.shape)

# x = np.array([])
x = []
for i in range(0, 1):  # int(360 / theta)):
    for j in range(0, 75):  # range(0, 4):
        # x = np.hstack([x, [i * theta]])
        x.append(i * theta)
print(x)

# temporarily disabling noise because of issues with pixel values
# A = A + y
clf = LinearDiscriminantAnalysis()
clf.fit(A, x)

z = img[200: 1000, 200: 1000].flatten()
print(clf.predict([z]))
z = rotate(img, 90)
z = z[200: 1000, 200: 1000].flatten()
print(clf.predict([z]))
z = rotate(img, 180)
z = z[200: 1000, 200: 1000].flatten()
print(clf.predict([z]))
z = rotate(img, 270)
z = z[200: 1000, 200: 1000].flatten()
print(clf.predict([z]))

# generating the test data
# for i in range(0, int(360 / 90)):  # 1 = 0, 90 = theta
#     new_img = rotate(img, 90 * i)  # 90 = theta
#     rot = new_img[200:1000, 200:1000].flatten()

#     200 + random.randint(0, 50)
#     U = new_img[200 + random.randint(0, 50):1000 + random.randint(0, 50), 200:1000].flatten()
#     D = new_img[100 + random.randint(0, 50):900 + random.randint(0, 50), 200:1000].flatten()
#     R = new_img[200:1000, 100 + random.randint(0, 50):900 + random.randint(0, 50)].flatten()
#     L = new_img[200:1000, 200 + random.randint(0, 50):1000 + random.randint(0, 50)].flatten()

#     C = compressor(new_img, 200, 1000, 200, 1000)
#     # [200:1000, 200:1000].flatten()
#     c = compressor(C, 200, 1000, 200, 1000)
#     # C = C[200:1000, 200:1000].flatten()

#     # witchcraft necessary for strong shifts at 4x compression. I need to
#     # think deliberately through these. make the base images large, and cut
#     # from them.
#     up = np.vstack([c, img[0:100, :]])
#     down = np.vstack([c[0:100, :], c])
#     right = np.hstack([c[:, 0:100], c[:, 0:700]])
#     left = np.hstack([c, img[:, 0:100]])

#     upright = np.hstack([up[:, 0:100], up])
#     upleft = np.hstack([up, up[:, 0:100]])
#     downright = np.hstack([c[0:800, 0:100], down[0:800, 0:700]])
#     downleft = np.hstack([down, up[:, 0:100]])

#     # UC, uC, Uc, uc, ... D.L.R (16)
#     U_C = C[400:1200, 200:1000].flatten()
#     uC = C[300:1100, 200:1000].flatten()
#     # Uc = c[500:1300, 200:1000].flatten() #***
#     Uc = up[500:1300, 200:1000].flatten()
#     uc = c[350:1150, 200:1000].flatten()

#     D_C = C[0:800, 200:1000].flatten()
#     dC = C[100:900, 200:1000].flatten()
#     Dc = down[0:800, 200:1000].flatten()
#     dc = c[50:850, 200:1000].flatten()

#     R_C = C[200:1000, 0:800].flatten()
#     rC = C[200:1000, 100:900].flatten()
#     Rc = right[200:1000, :].flatten()
#     # Rc = c[250:950, 0:700].flatten()
#     rc = c[200:1000, 50:850].flatten()

#     L_C = C[200:1000, 400:1200].flatten()
#     lC = C[200:1000, 300:1100].flatten()
#     # Lc = c[200:1000, 500:1300].flatten()  # ***
#     Lc = left[200:1000, 500:1300].flatten()
#     lc = c[200:1000, 350:1150].flatten()

#     # UL, uL, ul, Ul, UR, uR, ur, Ur, DL, dL, dl, Dl, DR, dR, dr, Dr (16)
#     U_L = new_img[250:1050, 250:1050].flatten()
#     uL = new_img[225:1025, 250:1050].flatten()
#     ul = new_img[225:1025, 225:1025].flatten()
#     Ul = new_img[250:1050, 225:1025].flatten()

#     U_R = new_img[250:1050, 150:950].flatten()
#     uR = new_img[225:1025, 150:950].flatten()
#     ur = new_img[225:1025, 175:975].flatten()
#     Ur = new_img[250:1050, 175:975].flatten()

#     D_L = new_img[150:950, 250:1050].flatten()
#     dL = new_img[175:975, 250:1050].flatten()
#     dl = new_img[175:975, 225:1025].flatten()
#     Dl = new_img[150:950, 225:1025].flatten()

#     D_R = new_img[150:950, 150:950].flatten()
#     dR = new_img[175:975, 150:950].flatten()
#     dr = new_img[175:975, 175:975].flatten()
#     Dr = new_img[150:950, 175:975].flatten()

#     # (UL uL ul Ul)(C, c), (UR, uR, ur, Ur)(C,c), ... (32)
#     # just named, none of the slices are accurate yet
#     U_L_C = C[400:1200, 400:1200].flatten()
#     uLC = C[300:1100, 400:1200].flatten()
#     ulC = C[300:1100, 300:1100].flatten()
#     UlC = C[400:1200, 300:1100].flatten()

#     U_R_C = C[400:1200, 0:800].flatten()
#     uRC = C[300:1100, 0:800].flatten()
#     urC = C[300:1100, 100:900].flatten()
#     UrC = C[400:1200, 100:900].flatten()

#     D_L_C = C[0:800, 400:1200].flatten()
#     dLC = C[100:900, 400:1200].flatten()
#     dlC = C[100:900, 300:1100].flatten()
#     DlC = C[0:800, 300:1100].flatten()

#     D_R_C = C[0:800, 0:800].flatten()
#     dRC = C[100:900, 0:800].flatten()
#     drC = C[100:900, 100:900].flatten()
#     DrC = C[0:800, 100:900].flatten()

#     # 2 shifts, 4x
#     # ULc = c[500:1300, 500:1300].flatten()  # ***
#     ULc = upleft[500:1300, 500:1300].flatten()
#     # uLc = c[350:1150, 500:1300].flatten()  # ***
#     uLc = left[350:1150, 500:1300].flatten()
#     ulc = c[350:1150, 350:1150].flatten()
#     # Ulc = c[500:1300, 350:1150].flatten()  # ***
#     Ulc = up[500:1300, 350:1150].flatten()

#     # URc = right[500:1300, :].flatten()  # *** np.hstack([img, img[:, 0:100]])
#     URc = upright[500:1300, 0:800].flatten()
#     uRc = right[350:1150, :].flatten()
#     urc = c[350:1150, 50:850].flatten()
#     # Urc = c[500:1300, 50:850].flatten()  # ***
#     Urc = up[500:1300, 50:850].flatten()

#     # DLc = down[:, 500:1300].flatten() #***
#     # np.hstack([down, img[:, 0:100]])[200:1000, 500:1300].flatten()
#     DLc = downleft[0:800, 500:1300].flatten()
#     dLc = left[50:850, 500:1300].flatten()  # ***
#     dlc = c[50:850, 350:1150].flatten()
#     Dlc = down[0:800, 350:1150].flatten()

#     DRc = downright.flatten()
#     dRc = right[50:850, :].flatten()
#     drc = c[50:850, 50:850].flatten()
#     Drc = down[0:800, 50:850].flatten()

#     C = C[200:1000, 200:1000].flatten()
#     c = c[200:1000, 200:1000].flatten()

#     if (i == 0):
#         A = np.vstack([[rot], [U], [D], [R], [L], [u], [d], [r], [l], [C], [c], [U_C], [uC], [Uc], [uc], [D_C], [dC], [Dc], [dc], [R_C], [rC], [Rc], [rc], [L_C], [lC], [Lc], [lc], [U_L], [uL], [ul], [Ul], [U_R], [uR], [ur], [Ur], [D_L], [dL], [dl], [Dl], [D_R], [dR], [dr],
#                        [Dr], [U_L_C], [uLC], [ulC], [UlC], [U_R_C], [uRC], [urC], [UrC], [D_L_C], [dLC], [dlC], [DlC], [D_R_C], [dRC], [drC], [DrC], [ULc], [uLc], [ulc], [Ulc], [URc], [uRc], [urc], [Urc], [DLc], [dLc], [dlc], [Dlc], [DRc], [dRc], [drc], [Drc]])
#         # A = np.array([rot, U, D, R, L, u, d, r, l, C, c, U_C, uC, Uc, uc, D_C, dC, Dc, dc, R_C, rC, Rc, rc, L_C, lC, Lc, lc, U_L, uL, ul, Ul, U_R, uR, ur, Ur, D_L, dL, dl, Dl, D_R, dR, dr, Dr, U_L_C,
#         # uLC, ulC, UlC, U_R_C, uRC, urC, UrC, D_L_C, dLC, dlC, DlC, D_R_C, dRC,
#         # drC, DrC, ULc, uLc, ulc, Ulc, URc, uRc, urc, Urc, DLc, dLc, dlc, Dlc,
#         # DRc, dRc, drc, Drc])  # compressed, compressed
#     else:
#         A = np.vstack([A, [rot], [U], [D], [R], [L], [u], [d], [r], [l], [C], [c], [U_C], [uC], [Uc], [uc], [D_C], [dC], [Dc], [dc], [R_C], [rC], [Rc], [rc], [L_C], [lC], [Lc], [lc], [U_L], [uL], [ul], [Ul], [U_R], [uR], [ur], [Ur], [D_L], [dL], [dl], [Dl], [D_R], [dR], [dr],
#                        [Dr], [U_L_C], [uLC], [ulC], [UlC], [U_R_C], [uRC], [urC], [UrC], [D_L_C], [dLC], [dlC], [DlC], [D_R_C], [dRC], [drC], [DrC], [ULc], [uLc], [ulc], [Ulc], [URc], [uRc], [urc], [Urc], [DLc], [dLc], [dlc], [Dlc], [DRc], [dRc], [drc], [Drc]])

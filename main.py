import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi


def ass_1():
    img = cv2.imread("rice.png", cv2.IMREAD_GRAYSCALE)

    filtered_img = ndi.median_filter(img, size=3)
    thresh = cv2.adaptiveThreshold(filtered_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,89,1)
    overlay = np.where(thresh > 0, 1, np.nan)

    plt.imshow(img, cmap='gray')
    plt.imshow(overlay, cmap='rainbow', alpha=0.70)
    plt.axis('off')
    plt.show()


def ass_2():
    img = cv2.imread("medtest.png", cv2.IMREAD_GRAYSCALE)

    mask = np.ones_like(img[:, :])
    height, width = mask.shape
    mask[int(3.37 * height / 5):, :] = 0

    blended_img = img.copy()
    blended_img[mask == 0] = 0

    ret, th = cv2.threshold(blended_img, 226, 255, cv2.THRESH_BINARY)
    th = cv2.blur(th, (3,5))
    mask_th = ndi.binary_closing(th, iterations=2)

    labels, nlabels = ndi.label(mask_th)

    overlay = np.where(labels > 0, labels, np.nan)

    plt.imshow(img, cmap='gray')
    plt.imshow(overlay, cmap='rainbow', alpha=0.7)
    plt.axis("off")
    plt.show()

ass_1()
import pickle
import random
from argparse import ArgumentParser
from functools import reduce
from pathlib import Path, PosixPath

import cv2 as cv
import torch
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch.models.mtcnn import fixed_image_standardization
from intercept import Intercept

resnet = InceptionResnetV1(pretrained='vggface2').eval()
interceptor = Intercept(resnet)

parser = ArgumentParser()
parser.add_argument('-f', '--folder', type=str)
parser.add_argument('-n', '--name', type=str)

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def spectrum(orig):
    orig_spectrum = np.fft.fftshift(np.fft.fft2(orig))
    return np.log(abs(orig_spectrum))

def draw_bbox_on_original_image(fmap, orig, color):

    cp = cv.cvtColor(orig.copy(), cv.COLOR_BGR2RGB)

    contours, _ = cv.findContours(fmap, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    prop_h = orig.shape[0] / fmap.shape[0]
    prop_w = orig.shape[1] / fmap.shape[1]

    for i, c in enumerate(contours):

        bbox_fmap = cv.boundingRect(c)
        box = np.uint0(bbox_fmap)

        (x0, y0, xf, yf) = int(box[0]), int(box[1]), int(box[0]+box[2]), int(box[1]+box[3])

        x1, y1, x2, y2 = int(x0 * prop_w), int(y0 * prop_h), int(xf * prop_w), int(yf * prop_h)
        cv.rectangle(cp, (x1, y1), (x2, y2), color, 2)

    return cp
    
def test_bbox_for_one_fmap(fmap, thresh, orig):

    f1 = cv.normalize(fmap, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
    _, f1 = cv.threshold(f1, thresh, 255, cv.THRESH_BINARY)

    cp = draw_bbox_on_original_image(f1, orig.copy(), random.sample(range(0, 255), 3))

    return cp

def apply_fourier(orig, modif):

    orig_spectrum = np.fft.fftshift(np.fft.fft2(orig))
    mod_spectrum = np.fft.fftshift(np.fft.fft2(modif))

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0, 0].imshow(orig, cmap='gray')
    ax[0, 1].imshow(modif, cmap='gray')
    ax[1, 0].imshow(np.log(abs(orig_spectrum)))
    ax[1, 1].imshow(np.log(abs(mod_spectrum)))

    plt.show()

def load_images(paths: list[PosixPath], png = False) -> list:
    images: list = []

    for path in paths:
        img = cv.imread('/'.join(path.parts))
        if png:
            img = np.dstack([img, img, img])
        images.append(img)

    return images

def load_images_and_label(mask_folder: Path, label=1):

    jpg_images: list[PosixPath] = list(mask_folder.glob("*.jpg"))
    jpg_images = load_images(jpg_images)

    jpeg_images: list[PosixPath] = list(mask_folder.glob("*.jpeg"))
    jpeg_images = load_images(jpeg_images)

    png_images: list[PosixPath] = list(mask_folder.glob("*.png"))
    png_images = load_images(png_images)

    images = png_images + jpeg_images + jpg_images
    images = np.array(images)
    labels = np.ones(images.shape[0])
    if label == 0:
        labels = np.zeros(images.shape[0])

    return images, labels

def load_features_and_labels(folder: Path):

    mask_folder: Path = folder / "Mask"
    images_1, labels_1 = load_images_and_label(mask_folder)

    mask_folder: Path = folder / "Non Mask"
    images_0, labels_0 = load_images_and_label(mask_folder, label=0)

    images = np.concatenate((images_0, images_1))
    labels = np.concatenate((labels_0, labels_1))

    return images, labels

print("[INFO] loading model...")
net = cv.dnn.readNetFromCaffe('/media/yves/HD5/Applications/vc_final/deploy.prototxt.txt', '/media/yves/HD5/Applications/vc_final/res10_300x300_ssd_iter_140000.caffemodel')

if __name__ == '__main__':

    args = parser.parse_args()
    path = Path(args.folder)

    images, labels = load_features_and_labels(path)
    c = list(zip(images, labels))

    random.shuffle(c)

    images, labels = zip(*c)

    BASE_PATH = Path('/'.join(path.parts[:-2])[1:] + "/faces")
    SAVE_MASK_PATH = BASE_PATH / path.parts[-1] / "Mask"
    SAVE_NON_MASK_PATH = BASE_PATH / path.parts[-1] / "Non Mask"

    i = 0
    for index, image in enumerate(images):
        (h, w) = image.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0,
	    (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()
        nice_cases = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:
                nice_cases.append(detections[0, 0, i, 3:7])

        nice_cases = np.array(nice_cases)
        for i in range(nice_cases.shape[0]):

            box = nice_cases[i, :] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if startY > endY:
                startY, endY = endY, startY
            
            if startX > endX:
                startX, endX = endX, startX
        
            crop = image[startY: endY, startX: endX, :]

            if reduce(lambda a, b: a * b, crop.shape) == 0:
                continue

            if labels[index] == 1:
                cv.imwrite(str(SAVE_MASK_PATH / "{}.jpg".format(i)), crop)
            if labels[index] == 0:
                cv.imwrite(str(SAVE_NON_MASK_PATH / "{}.jpg".format(i)), crop)
            i += 1


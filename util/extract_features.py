import pickle
import random
from argparse import ArgumentParser
from pathlib import Path, PosixPath
from sklearn.tree import DecisionTreeClassifier
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

def load_images_and_labels(folder: Path):

    mask_folder: Path = folder / "Mask"
    images_1, labels_1 = load_images_and_label(mask_folder)

    mask_folder: Path = folder / "Non Mask"
    images_0, labels_0 = load_images_and_label(mask_folder, label=0)

    tam_mask = len(images_1)
    f = int(0.8 * tam_mask)

    x_train_mask, x_test_mask = images_1[0:f], images_1[f:]
    y_train_mask, y_test_mask = labels_1[0:f], labels_1[f:]

    tam_non_mask = len(images_0)
    f = int(0.8 * tam_non_mask)

    x_train_non_mask, x_test_non_mask = images_0[0:f], images_0[f:]
    y_train_non_mask, y_test_non_mask = labels_0[0:f], labels_0[f:]


    Xtrain, Xtest = np.concatenate((x_train_mask, x_train_non_mask)), np.concatenate((x_test_mask, x_test_non_mask))
    Ytrain, Ytest = np.concatenate((y_train_mask, y_train_non_mask)), np.concatenate((y_test_mask, y_test_non_mask))
    
    return Xtrain, Ytrain, Xtest, Ytest

if __name__ == '__main__':

    args = parser.parse_args()
    path = Path(args.folder)

    (Xtrain, Ytrain, Xtest, Ytest) = load_images_and_labels(path)
    c = list(zip(Xtrain, Ytrain))

    random.shuffle(c)

    Xtrain, Ytrain = zip(*c)

    x_train, y_train = [], []
    print("[TRAIN] ------------------ MAKE FEATURES ------------------")
    for index, image in enumerate(Xtrain):
        image = cv.resize(image, (160, 160), cv.INTER_AREA)
        tensor = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
        output = interceptor.forward(tensor)

        fmap = output['conv2d_4a'].detach().numpy()[0][143]
        x_train.append(fmap)
        y_train.append(Ytrain[index])
        print("[{}/{}] Extracting features...".format(index + 1, len(Xtrain)))
    print()
    print()
    print("[TEST] ------------------ MAKE FEATURES ------------------")
    x_test, y_test = [], []
    for index, image in enumerate(Xtest):
        image = cv.resize(image, (160, 160), cv.INTER_AREA)
        tensor = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
        output = interceptor.forward(tensor)

        fmap = output['conv2d_4a'].detach().numpy()[0][143]
        x_test.append(fmap)
        y_test.append(Ytest[index])
        print("[{}/{}] Extracting features...".format(index + 1, len(Xtest)))

        # fix, ax = plt.subplots(16, 12)

        # k = 0
        # for i in range(16):
        #     for j in range(12):
        #         ax[i,j].imshow(fmaps1[k])
        #         ax[i,j].set_xlabel(Ytrain[index])
        #         k += 1
        # print("Label: {}".format(Ytrain[index]))
        # plt.imshow(fmaps1[143])
        # plt.show()

        # conv2d_3b -> 15
        # conv2d_4b -> (16 * k - 1), onde k = 9, 10, 11, 12

    train_data = {"features": x_train, "target": y_train}
    with open(args.name + "_train.pickle", "wb") as file:
        pickle.dump(train_data, file, protocol=pickle.HIGHEST_PROTOCOL)

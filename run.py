import glob, os, torch
import cv2
import numpy as np
from depth_anything_3.api import DepthAnything3
from img_processing import array_info


def da3_model_initial():
    """
    Initialize the Depth-anything 3 model.
    :return: Depth-anything 3 model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained('depth-anything/da3metric-large')
    model = model.to(device)
    return model

def predict_raw_depth(model, img):
    """
    Predict raw image with a trained Depth-anything 3 model.
    :param model: Depth-anything 3 model
    :param img: (numpy array) input image
    :return: (numpy array, float) predicted raw depth, depth unit : m
    """
    prediction = model.inference([img])
    # Re-size to the input resolution
    raw_depth = cv2.resize(prediction.depth[0],(640,480),interpolation=cv2.INTER_LINEAR)
    return raw_depth

if __name__ == '__main__':
    # Initialize the DA3 model
    model = da3_model_initial()

    # Load images
    img_0 = cv2.imread('test_img/G_00_rgb.png', cv2.COLOR_BGRA2BGR)
    print('G_00_rgb : ' + array_info(img_0))

    img_1 = cv2.imread('test_img/G_05_rgb.png', cv2.COLOR_BGRA2BGR)
    print('G_05_rgb : ' + array_info(img_1))

    img_2 = cv2.imread('test_img/G_10_rgb.png', cv2.COLOR_BGRA2BGR)
    print('G_10_rgb : ' + array_info(img_2))

    img_3 = cv2.imread('test_img/G_19_rgb.png', cv2.COLOR_BGRA2BGR)
    print('G_19_rgb : ' + array_info(img_3))

    img_4 = cv2.imread('test_img/M1_01_intensity_darken05.png', cv2.IMREAD_UNCHANGED)
    print('M1_01_intensity_darken05 : ' + array_info(img_4))

    img_5 = cv2.imread('test_img/M1_02_intensity_image.png', cv2.IMREAD_GRAYSCALE)
    print('M1_02_intensity_image : ' + array_info(img_5))

    img_6 = cv2.imread('test_img/M1_05_intensity_image.png', cv2.IMREAD_GRAYSCALE)
    print('M1_05_intensity_image : ' + array_info(img_6))

    img_7 =cv2.imread('test_img/M1_09_intensity_image.png', cv2.IMREAD_GRAYSCALE)
    print('M1_09_intensity_image.png : ' + array_info(img_7))

    img_8 = cv2.imread('test_img/M1_13_intensity_image.png', cv2.IMREAD_GRAYSCALE)
    print('M1_13_intensity_image.png : ' + array_info(img_8))

    predict_depth = predict_raw_depth(model, img_8)
    print('predict_depth : ' + array_info(predict_depth))
    np.save('result_img/M1_13_intensity_image_predict', predict_depth)
    cv2.imshow('prediction_img', predict_depth)
    cv2.waitKey(0)

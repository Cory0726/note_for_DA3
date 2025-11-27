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

def main():
    # Initialize the DA3 model
    model = da3_model_initial()

    img = cv2.imread('test_img/M1_08_intensity_image.png', cv2.IMREAD_GRAYSCALE)
    print(array_info(img))

    # Predict raw depth
    predict_depth = predict_raw_depth(model, img)
    print('predict_depth : ' + array_info(predict_depth))
    # Save as .npy
    np.save('result_img/M1_08_predict_depth', predict_depth)

if __name__ == '__main__':
   # real_depth = np.load('test_img/M1_01_raw_depth.npy')
   # print('real depth : ' + array_info(real_depth))
   # predict_depth = np.load('result_img/M1_01_predict_depth.npy')
   # print('predict_depth : ' + array_info(predict_depth))
   main()

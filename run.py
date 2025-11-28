import glob, os, torch
import cv2
import numpy as np
from depth_anything_3.api import DepthAnything3
from img_processing import array_info
from sklearn.linear_model import LinearRegression, RANSACRegressor

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

def calibrate_depth_ransac(
    D_da3,                      # DA3 depth map, shape (H, W), meter
    D_tof,                      # ToF depth map, shape (H, W), meter
    mask,                       # Mask image, 255 = use pixel, 0 = ignore
    min_depth=0.2,             # Minimum valid depth (meter)
    max_depth=1.5,              # Maximum valid depth (meter)
    residual_threshold=0.02,    # RANSAC inlier threshold (meter)
    min_samples=200,            # Minimum valid samples for calibration
    max_trials=100              # RANSAC iterations
):
    """
    Calibrate DA3 depth using ToF depth in masked regions (mask = 255).
    Uses RANSAC to robustly fit:
        d_tof ≈ a * d_da3 + b

    Returns:
        D_da3_calibrated : np.ndarray (H, W), calibrated DA3 depth (meter)
    """

    # Build valid-pixel mask
    valid = (
        (mask == 255) &
        np.isfinite(D_da3) & np.isfinite(D_tof) &  # Remove NaNs
        (D_da3 > min_depth) & (D_da3 < max_depth) &
        (D_tof > min_depth) & (D_tof < max_depth)
    )
    cv2.imwrite('img_process_temp/valid_pixel_mask.png', valid.astype(np.uint8) * 255)

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
    # Load the ToF raw depth
    tof_depth = (np.load('test_img/M1_01_raw_depth.npy') / 1000)  # Unit : m
    print('ToF depth : ' + array_info(tof_depth))
    # Load the predicted depth
    predict_depth = np.load('result_img/M1_01_predict_depth.npy')  # Unit : m
    print('Predict depth : ' + array_info(predict_depth))
    # Load the mask of hand segmentation
    hand_seg_mask = cv2.imread('test_img/M1_01_mask.png', cv2.IMREAD_UNCHANGED)  # [255, 0]
    print('Hand Seg mask : ' + array_info(hand_seg_mask))

    # Calibrate the preditct depth to real depth
    calibrate_depth_ransac(predict_depth,real_depth,hand_seg_mask)
   # main()

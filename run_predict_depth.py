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

def predict_da3_depth(da3_model, img):
    """
    Predict raw image with a trained Depth-anything 3 model.
    :param da3_model: Depth-anything 3 model
    :param img: (numpy array) input image
    :return: (numpy array, float) predicted raw depth, depth unit : m
    """
    prediction = da3_model.inference([img])
    # Re-size to the input resolution
    raw_depth = cv2.resize(prediction.depth[0],(640,480),interpolation=cv2.INTER_LINEAR)
    return raw_depth

def calibrate_depth_ransac(
    D_da3,                      # DA3 depth map, shape (H, W), meter
    D_tof,                      # ToF depth map, shape (H, W), meter
    mask,                       # Mask image, 255 = use pixel, 0 = ignore
    min_depth=0.2,             # Minimum valid depth (meter)
    max_depth=0.5,              # Maximum valid depth (meter)
    residual_threshold=0.01,    # RANSAC inlier threshold (meter)
    min_samples=20000,            # Minimum valid samples for calibration
    max_trials=1000              # RANSAC iterations
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
    # Check the number of valid pixels
    print('Number of valid pixels: ', valid.sum())
    if valid.sum() < min_samples:
        print('[WARNING] Not enough valid pixels for RANSAC calibration. Skipping calibration.')
        return D_da3.copy()
    # Extract valid pixels >> flatten to 1D arrays
    d_da3 = D_da3[valid].astype(np.float32)  # shape (N,)
    d_tof = D_tof[valid].astype(np.float32)  # shape (N,)
    # RANSAC robust regression
    ransac_model = RANSACRegressor(
        LinearRegression(),
        max_trials=max_trials,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
    )
    ransac_model.fit(d_da3.reshape(-1, 1), d_tof)
    # Extract scale and offset
    a = float(ransac_model.estimator_.coef_[0])
    b = float(ransac_model.estimator_.intercept_)
    print(f"[Calibration] d_tof ≈ {a:.4f} * d_da3 + {b:.4f}")
    # Apply calibration to the entire predicted depth map of DA3
    D_da3_calibrated = a * D_da3 + b
    return D_da3_calibrated


def main(input_img_file, tof_depth_file, hand_seg_mask_file, output_depth_file):
    """
    Predict Depth-anything 3 depth with RANSAC calibration.
    :param input_img_file: Path to grayscale or intensity input image for DA3 prediction.
    :param tof_depth_file: Path to .file containing tof depth in millimeters.
    :param hand_seg_mask_file: Path to segmentation mask (uint8 image with values {255 = hand, 0 = background}).
    :param output_depth_file: Output path (.npy) for saving the calibrated depth map (meters).
    """
    # ==================================================
    # Predict the raw depth by Depth-Anything-3 model
    # ==================================================
    # Initialize the DA3 model
    model = da3_model_initial()
    # Load original image for DA3 model predict
    img = cv2.imread(input_img_file, cv2.IMREAD_GRAYSCALE)
    print('Original image for DA3 predict : ' + array_info(img))
    # Predicted raw depth
    predicted_depth = predict_da3_depth(model, img)
    print('Predicted raw depth by DA3 model : ' + array_info(predicted_depth))
    # Save the predicted raw depth as .npy
    np.save('img_process_temp/predicted_raw_depth', predicted_depth)

    # ==================================================
    # Calibrate the predicted raw depth with ToF raw depth and Hand Segmentation mask
    # ==================================================
    # Load the ToF raw depth
    tof_depth = (np.load(tof_depth_file) / 1000)  # Unit : m
    print('ToF raw depth : ' + array_info(tof_depth))
    # Load the mask of hand segmentation
    hand_seg_mask = cv2.imread(hand_seg_mask_file, cv2.IMREAD_UNCHANGED)  # [255, 0]
    print('Hand Seg mask : ' + array_info(hand_seg_mask))
    # Calibrate the predicted raw depth
    calibrate_depth = calibrate_depth_ransac(predicted_depth, tof_depth, hand_seg_mask)
    print('Calibrated depth : ' + array_info(calibrate_depth))
    # Save the calibrated depth as .npy
    np.save(output_depth_file, calibrate_depth)

if __name__ == '__main__':
    main(
        input_img_file='test_img/M1_01_intensity_image.png',
        tof_depth_file='test_img/M1_01_raw_depth.npy',
        hand_seg_mask_file='test_img/M1_01_mask.png',
        output_depth_file='result_img/M1_01_predicted_depth_with_calibration.npy'
    )

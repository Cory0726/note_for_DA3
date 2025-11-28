import numpy as np
import cv2

def inspect_depth_pair(input_depth_1, input_depth_2):
    """
    Visualize two depth maps (.npy) and inspect values by clicking on input_1.

    - Left window: input_1
    - Right window: input_2
    - Click on input_1 window to:
        * print (x, y)
        * print depth1[y, x] and depth2[y, x]
        * draw a red dot on both images at the clicked position

    Press 'q' or 'Esc' to exit.
    """

    # -------------------------
    # Load depth maps from .npy
    # -------------------------
    depth1 = np.load(input_depth_1)  # shape (H, W)
    depth2 = np.load(input_depth_2)  # shape (H, W)

    if depth1.shape != depth2.shape:
        raise ValueError(f"Depth maps must have the same shape, "
                         f"got {depth1.shape} and {depth2.shape}")

    # -------------------------
    # Prepare visualization images
    # -------------------------
    def depth_to_color(depth):
        """Normalize depth to 0~255 and apply a colormap for visualization."""
        depth_vis = depth.copy()

        # Handle NaN / inf
        depth_vis = np.where(np.isfinite(depth_vis), depth_vis, 0)

        # Use percentiles to avoid extreme outliers affecting contrast
        d_min = np.percentile(depth_vis, 1)
        d_max = np.percentile(depth_vis, 99)

        if d_max <= d_min:  # fallback
            d_min = float(depth_vis.min())
            d_max = float(depth_vis.max())

        if d_max == d_min:
            d_max = d_min + 1e-6

        depth_norm = np.clip((depth_vis - d_min) / (d_max - d_min), 0, 1)
        depth_8u = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        return depth_color

    vis1 = depth_to_color(depth1)
    vis2 = depth_to_color(depth2)

    # state dict to share between callback and main loop
    state = {
        "depth1": depth1,
        "depth2": depth2,
        "vis1": vis1,
        "vis2": vis2,
    }

    # -------------------------
    # Mouse callback
    # -------------------------
    def on_mouse_input1(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        d1 = state["depth1"]
        d2 = state["depth2"]

        h, w = d1.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        val1 = d1[y, x]
        val2 = d2[y, x]

        print(f"Clicked at (x={x}, y={y})")
        print(f"  input_1 depth: {val1}")
        print(f"  input_2 depth: {val2}")

        # Draw points on both visualization images
        cv2.circle(state["vis1"], (x, y), 3, (0, 0, 255), -1)  # red dot
        cv2.circle(state["vis2"], (x, y), 3, (0, 0, 255), -1)

    # -------------------------
    # Create windows and set callback
    # -------------------------
    cv2.namedWindow("input_1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("input_2", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("input_1", on_mouse_input1)

    print("Instructions:")
    print(" - Click on 'input_1' window to inspect depth values.")
    print(" - Press 'q' or 'Esc' to exit.")

    # -------------------------
    # Main loop
    # -------------------------
    while True:
        cv2.imshow("input_1", state["vis1"])
        cv2.imshow("input_2", state["vis2"])

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q')):  # 27 = Esc
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    inspect_depth_pair(
        input_depth_1='test_img/M1_08_raw_depth.npy',
        input_depth_2='result_img/M1_08_predicted_depth_with_calibration_method5.npy',
    )

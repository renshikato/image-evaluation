# utils/color.py
import cv2 as cv
import numpy as np

def apply_gray_world_wb(img_bgr):
    """
    グレーワールド法を用いてホワイトバランスを調整する。
    """
    img_float = img_bgr.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(img_float[:, :, 0]), np.mean(img_float[:, :, 1]), np.mean(img_float[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    scales = [avg_gray / a if a > 0 else 1.0 for a in [avg_b, avg_g, avg_r]]
    
    for i in range(3):
        img_float[:, :, i] *= scales[i]
        
    max_val = 255 if img_bgr.dtype == np.uint8 else 65535
    return np.clip(img_float, 0, max_val).astype(img_bgr.dtype)

def apply_lab_color_correction(img_bgr_16bit):
    """
    LAB空間を利用して色かぶりを自動補正する（16bit画像用）。
    """
    img_float = img_bgr_16bit.astype(np.float32) / 65535.0
    img_lab = cv.cvtColor(img_float, cv.COLOR_BGR2Lab)
    l, a, b = cv.split(img_lab)
    
    # A, Bチャネルの平均値を0に近づける
    a = a - np.mean(a)
    b = b - np.mean(b)
    
    img_lab_corrected = cv.merge([l, a, b])
    img_bgr_corrected = cv.cvtColor(img_lab_corrected, cv.COLOR_Lab2BGR)
    
    return (np.clip(img_bgr_corrected, 0.0, 1.0) * 65535.0).astype(np.uint16)
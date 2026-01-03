# utils/processing.py
import cv2 as cv

def apply_marine_snow_removal(img, kernel_size, threshold, inpaint_radius, return_mask=False):
    """
    トップハット変換によるノイズ検出と、インペイントによるノイズ除去を実行する。
    """
    # 1. ノイズ検出 (トップハット変換)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    
    # 2. マスク作成と膨張
    _, mask = cv.threshold(tophat, threshold, 255, cv.THRESH_BINARY)
    mask = cv.dilate(mask, None, iterations=1)

    if return_mask:
        return mask

    # 3. インペイントによる修復
    result = cv.inpaint(img, mask, inpaint_radius, cv.INPAINT_TELEA)
    return result
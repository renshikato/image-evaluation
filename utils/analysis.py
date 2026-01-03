# utils/analysis.py
import cv2 as cv
import numpy as np

def get_rgb_histogram(img):
    """
    RGB各チャネルのヒストグラムを計算し、全チャネルの最大頻度で正規化して返す。
    """
    hists = []
    for i in range(3):
        # 各チャネル(B, G, R)のヒストグラムを計算
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        hists.append(hist)
    
    # 全チャネルを通じた最大頻度を取得し、ゼロ除算を防止
    global_max = max([h.max() for h in hists])
    if global_max == 0:
        global_max = 1.0
        
    return [h / global_max for h in hists]

def get_sv_distribution(img_bgr):
    """
    BGR画像からHSVのSaturationとValueの分布データを取得する。
    """
    img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    _, s, v = cv.split(img_hsv)
    return s.flatten(), v.flatten()

def calculate_fft_magnitude(gray_img):
    """
    グレースケール画像からFFT（高速フーリエ変換）による振幅スペクトルを計算する。
    """
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    # 対数スケールで振幅スペクトルを計算
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum
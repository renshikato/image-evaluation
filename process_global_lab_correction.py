"""
フォトグラメトリー向け前処理: LAB Color Correction + Global Linear Norm

1. [LAB色補正] 画像をLAB空間に変換し、A, Bチャネル(色味)の分布中心を0に合わせることで、
   緑かぶりなどの色偏りを自動補正する。(グレーワールド法のような過剰な赤浮きを抑制)
2. [Global正規化] データセット全体のMin/Maxを用いて正規化。

実行方法:
    python3 process_lab_correction.py <入力フォルダ> <出力親フォルダ>
"""

import glob
import os
import sys

import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 設定 ---
plt.rcParams.update({'font.size': 14})
Y_AXIS_MAX_RATIO = 1.0
GLOBAL_CLIP_LOW = 0.0
GLOBAL_CLIP_HIGH = 100.0
# -----------

def apply_lab_color_correction(img_bgr_16bit: np.ndarray) -> np.ndarray:
    """
    LAB色空間を利用して自動的に色かぶりを除去する。
    16bit画像を float32 に変換して計算し、16bitに戻す。
    """
    if len(img_bgr_16bit.shape) < 3: return img_bgr_16bit

    # 1. 計算用にfloat32 (0.0-1.0) に正規化
    img_float = img_bgr_16bit.astype(np.float32) / 65535.0
    
    # 2. BGR -> LAB 変換
    img_lab = cv.cvtColor(img_float, cv.COLOR_BGR2Lab)
    
    # L, A, B に分離
    l, a, b = cv.split(img_lab)
    
    # 3. 色かぶり補正 (A, Bチャネルの中心を "128相当" に戻す)
    # OpenCVのfloat LABでは、L:0-100, A:-127~127, B:-127~127 ではなく
    # 実装依存の値になることがあるため、平均値のシフトで対応する。
    
    # A, Bチャネルの平均値を計算
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    
    # 本来、無彩色の平均は 0 (OpenCVのfloat変換の実装によっては128バイアスがない場合がある)
    # ここでは「平均値を0(無彩色)に近づける」シフトを行う
    # ※注: OpenCVの cvtColor(..., BGR2Lab) float32出力の仕様:
    # L, A, B はスケーリングされない生のCIE Lab値に近い値を持つか、
    # あるいは 0-1 などの範囲に収まるかはフラグやバージョンによるが、
    # 一般的に相対的なズレを補正すればよい。
    
    # 単純に「平均値を引く」ことで分布の中心を0に寄せたいが、
    # OpenCVのLab変換(float)の仕様に合わせるため、
    # 「今の平均」と「理想の平均(色味なし)」の差分を引くアプローチをとる。
    # ここではシンプルに「AとBの平均値を引いて、無彩色点(0付近)に持っていく」処理とする。
    
    # Aチャネルの補正 (緑-赤)
    a = a - ((mean_a - 0) * 1.0) # 1.0は補正強度。弱めるなら0.5とかにする
    
    # Bチャネルの補正 (青-黄)
    b = b - ((mean_b - 0) * 1.0)
    
    # 4. 結合して BGR に戻す
    img_lab_corrected = cv.merge([l, a, b])
    img_bgr_corrected = cv.cvtColor(img_lab_corrected, cv.COLOR_Lab2BGR)
    
    # 5. 範囲クリップして 16bit に戻す
    img_bgr_corrected = np.clip(img_bgr_corrected, 0.0, 1.0)
    return (img_bgr_corrected * 65535.0).astype(np.uint16)


def find_global_stats_with_lab(image_files):
    print(f"[Phase 1] Global統計の算出 (LAB補正後)...")
    vmins, vmaxs, valid_files = [], [], []
    
    for i, fpath in enumerate(image_files):
        if i % 10 == 0: print(f"\rScanning: {i}/{len(image_files)}", end="")
        img = cv.imread(fpath, cv.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint16: continue
        valid_files.append(fpath)
        
        # LAB補正をかけた状態でMin/Maxを計測
        img_corrected = apply_lab_color_correction(img)
        
        vmins.append(np.min(img_corrected))
        vmaxs.append(np.max(img_corrected))
    
    print(f"\rScanning: {len(image_files)}/{len(image_files)} [Done]")
    if not vmins: return None, None, []

    global_vmin = np.percentile(vmins, GLOBAL_CLIP_LOW)
    global_vmax = np.percentile(vmaxs, GLOBAL_CLIP_HIGH)
    print(f"\n[RESULT] Global Range: {global_vmin:.1f} - {global_vmax:.1f}")
    return global_vmin, global_vmax, valid_files

def convert_to_8bit_global(img_16bit, vmin, vmax):
    if vmin >= vmax: return np.zeros_like(img_16bit, dtype=np.uint8)
    img_clipped = np.clip(img_16bit, vmin, vmax)
    img_norm = (img_clipped.astype(np.float32) - vmin) / (vmax - vmin)
    return (img_norm * 255.0).astype(np.uint8)

# --- Plotting (RGB相対比版) ---
def get_display_image(image_raw):
    if len(image_raw.shape) == 3 and image_raw.shape[2] >= 3:
        disp = cv.cvtColor(image_raw[:, :, :3], cv.COLOR_BGR2RGB)
    else: disp = image_raw
    if disp.dtype == np.uint16: return (disp >> 8).astype(np.uint8)
    return disp

def plot_rgb_hist(ax, img, title, max_v):
    bins=256
    channels = cv.split(img[:,:,:3])
    colors = ('blue', 'green', 'red')
    peak = 0
    for c in channels:
        cnt, _ = np.histogram(c, bins, [0, max_v])
        if cnt.max() > peak: peak = cnt.max()
    if peak == 0: peak = 1
    
    for c, col, lbl in zip(channels, colors, ('B','G','R')):
        w = np.ones_like(c.ravel()) / peak
        ax.hist(c.ravel(), bins, [0, max_v], color=col, alpha=0.6, weights=w, label=lbl)
    ax.set_title(title); ax.set_xlim(0, max_v); ax.set_ylim(0, 1.0); 
    ax.set_xlabel("Value"); ax.set_ylabel("Norm Count (Max=1.0)"); ax.legend(loc='upper right')

def plot_comparison(img_orig, img_res, path, name):
    disp_orig = get_display_image(img_orig)
    disp_res = get_display_image(img_res)
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f"LAB Correction & Global Norm\n{name}", fontsize=24)
    
    ax1=plt.subplot(2,2,1); ax1.imshow(disp_orig); ax1.set_title("Original (16bit)"); ax1.axis('off')
    ax2=plt.subplot(2,2,2); plot_rgb_hist(ax2, img_orig, "Original RGB Hist", 65536)
    ax3=plt.subplot(2,2,3); ax3.imshow(disp_res); ax3.set_title("Result (LAB Corrected + 8bit)"); ax3.axis('off')
    ax4=plt.subplot(2,2,4); plot_rgb_hist(ax4, img_res, "Result RGB Hist", 256)
    
    plt.tight_layout()
    try: plt.savefig(path); plt.close(fig)
    except: plt.close(fig)

# --- Main ---
def process_folder(input_dir, output_parent):
    if not os.path.isdir(input_dir): return
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.[tT][iI][fF]")) + 
                         glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]")))
    if not image_files: return

    # 1. Global Range 算出 (LAB補正後の値に基づく)
    g_vmin, g_vmax, valid_files = find_global_stats_with_lab(image_files)
    if g_vmin is None: return
    
    # 2. 出力
    suffix = "LAB_AutoCorrect_GlobalLinear"
    dir_img = os.path.join(output_parent, f"img_{suffix}")
    dir_hist = os.path.join(output_parent, f"hist_{suffix}")
    os.makedirs(dir_img, exist_ok=True)
    os.makedirs(dir_hist, exist_ok=True)
    
    print(f"[INFO] 画像出力先: {dir_img}")

    # 3. 変換
    for fpath in valid_files:
        name = os.path.splitext(os.path.basename(fpath))[0]
        try:
            img_16 = cv.imread(fpath, cv.IMREAD_UNCHANGED)
            
            # LAB補正
            img_lab_corr = apply_lab_color_correction(img_16)
            
            # Global正規化 (8bit化)
            img_8 = convert_to_8bit_global(img_lab_corr, g_vmin, g_vmax)
            
            cv.imwrite(os.path.join(dir_img, f"{name}.png"), img_8)
            plot_comparison(img_16, img_8, os.path.join(dir_hist, f"{name}_comp.png"), name)
            
        except Exception as e:
            print(f"Error {name}: {e}")

    print("\n[INFO] 完了。")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python3 process_lab_correction.py <input_dir> <output_parent>")
        sys.exit(1)
    process_folder(sys.argv[1], sys.argv[2])
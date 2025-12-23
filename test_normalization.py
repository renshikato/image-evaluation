"""
フォトグラメトリー向け前処理: Global Linear Normalization

特徴:
1. [Global統計] 全画像から「絶対的な」最小値(0%)と最大値(100%)を取得。
   これにより、どんなに明るい画像でも白飛びさせない。
2. [線形変換] ガンマ補正を行わず(Gamma=1.0)、ヒストグラムを素直に引き伸ばす。
   明るい部分の階調圧縮を防ぐ。
3. [WBなし] 色味の変更は行わない（必要であれば前段で行う想定）。

実行方法:
    python3 process_global_linear.py <入力フォルダ> <出力親フォルダ>
"""

import glob
import os
import sys
import random

import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 設定 ---
plt.rcParams.update({'font.size': 14})
Y_AXIS_MAX_RATIO = 1.0

# --- ユーザー設定パラメータ ---
# 絶対的な範囲を取得するため 0.0 と 100.0 に設定
GLOBAL_CLIP_LOW = 0.0
GLOBAL_CLIP_HIGH = 100.0

# 線形変換 (ヒストグラムを歪めない)
GAMMA_VAL = 1.0

# 統計用サンプリング率
SAMPLE_RATE = 0.01
# ---------------------------

def find_global_stats(image_files):
    print(f"[Phase 1] Global統計の算出 (Linear / Full Range)...")
    
    # 高速化: 全画像を舐めるが、極端な外れ値（ホットピクセル）対策として
    # 各画像の min/max を取得し、その集合から決定する
    vmins = []
    vmaxs = []
    valid_files = []
    
    for i, fpath in enumerate(image_files):
        if i % 10 == 0: print(f"\rScanning: {i}/{len(image_files)}", end="")
        
        img = cv.imread(fpath, cv.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint16: continue
        
        valid_files.append(fpath)
        
        # この画像の最小・最大を取得
        # (ノイズが気になる場合はここで 0.1 / 99.9 程度にしても良いが、
        #  白飛び回避最優先なら min/max を使う)
        vmins.append(np.min(img))
        vmaxs.append(np.max(img))
    
    print(f"\rScanning: {len(image_files)}/{len(image_files)} [Done]")
    
    if not vmins:
        print("[ERROR] 有効な画像がありません。")
        return None, None, []

    # データセット全体での最小・最大
    # percentle(0) は min と同じ、percentile(100) は max と同じ
    global_vmin = np.percentile(vmins, GLOBAL_CLIP_LOW)
    global_vmax = np.percentile(vmaxs, GLOBAL_CLIP_HIGH)
    
    print(f"\n[RESULT] Global Range: {global_vmin:.1f} - {global_vmax:.1f}")
    return global_vmin, global_vmax, valid_files

def convert_with_gamma_global(img_16bit, vmin, vmax, gamma):
    """Global Range と ガンマ補正(1.0) を用いて8bit化"""
    img_clipped = np.clip(img_16bit, vmin, vmax)
    img_norm = (img_clipped.astype(np.float32) - vmin) / (vmax - vmin)
    # ガンマ適用 (1.0の場合は変化なし)
    if gamma != 1.0:
        img_norm = np.power(img_norm, gamma)
    return (img_norm * 255.0).astype(np.uint8)

# --- プロット関数 ---
def get_gray_image(image_raw):
    if len(image_raw.shape) == 3 and image_raw.shape[2] >= 3:
        if image_raw.dtype == np.uint8: return cv.cvtColor(image_raw, cv.COLOR_BGR2GRAY)
        else: return image_raw[:, :, 1]
    return image_raw

def get_display_image(image_raw):
    if len(image_raw.shape) == 3 and image_raw.shape[2] >= 3:
        disp = cv.cvtColor(image_raw[:, :, :3], cv.COLOR_BGR2RGB)
    else: disp = image_raw
    if disp.dtype == np.uint16: return (disp >> 8).astype(np.uint8)
    return disp

def plot_lum_hist(ax, img, title):
    gray = get_gray_image(img)
    if gray.dtype == np.uint8: max_v = 256; xl = "Lum (8bit)"
    else: max_v = 65536; xl = "Lum (16bit)"
    
    bins = 256
    counts, _ = np.histogram(gray, bins=bins, range=[0, max_v])
    peak = counts.max() if counts.max() > 0 else 1.0
    w = np.ones_like(gray.ravel()) / peak

    ax.hist(gray.ravel(), bins=bins, range=[0, max_v], 
            color='black', edgecolor='gray', alpha=0.7, 
            weights=w, label='Luminance')
    ax.set_title(title); ax.set_xlim(0, max_v); ax.set_ylim(0, 1.0)
    ax.set_xlabel(xl); ax.set_ylabel("Normalized Count(Max=1.0)")

def plot_comparison(img_orig, img_res, path, name, param_str):
    disp_orig = get_display_image(img_orig)
    disp_res = get_display_image(img_res)
    
    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f"Global Linear Norm\n{name}", fontsize=24)
    
    ax1=plt.subplot(2,2,1); ax1.imshow(disp_orig); ax1.set_title("Original (16bit)"); ax1.axis('off')
    ax2=plt.subplot(2,2,2); plot_lum_hist(ax2, img_orig, "Original Luminance Hist")
    ax3=plt.subplot(2,2,3); ax3.imshow(disp_res); ax3.set_title(f"Result (8bit)\n{param_str}"); ax3.axis('off')
    ax4=plt.subplot(2,2,4); plot_lum_hist(ax4, img_res, "Result Luminance Hist")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try: plt.savefig(path); plt.close(fig)
    except: plt.close(fig)

# --- Main ---
def process_folder(input_dir, output_parent):
    if not os.path.isdir(input_dir): return
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.[tT][iI][fF]")) + 
                         glob.glob(os.path.join(input_dir, "*.[pP][nN][gG]")))
    if not image_files: return

    # 1. Global Range (Min=0%, Max=100%)
    g_vmin, g_vmax, valid_files = find_global_stats(image_files)
    if g_vmin is None: return
    
    # 2. 出力
    suffix = f"GlobalLinear_Min{int(g_vmin)}_Max{int(g_vmax)}"
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
            
            # 線形変換 (Gamma=1.0)
            img_8 = convert_with_gamma_global(img_16, g_vmin, g_vmax, GAMMA_VAL)
            
            cv.imwrite(os.path.join(dir_img, f"{name}.png"), img_8)
            
            info_str = f"Range: {g_vmin:.0f}-{g_vmax:.0f}, Gamma: {GAMMA_VAL:.1f}"
            plot_comparison(img_16, img_8, os.path.join(dir_hist, f"{name}_comp.png"), name, info_str)
            
        except Exception as e:
            print(f"Error {name}: {e}")

    print("\n[INFO] 完了。")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python3 process_global_linear.py <input_dir> <output_parent>")
        sys.exit(1)
    process_folder(sys.argv[1], sys.argv[2])
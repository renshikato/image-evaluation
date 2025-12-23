import sys
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def analyze_and_save(image_path, graph_output_dir, map_output_dir):
    """
    1枚の画像を解析し、結果をoutput_dirに保存する
    """
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    print(f"Processing: {filename} ...")

    img = cv.imread(image_path)
    if img is None:
        print(f"  -> 読み込み失敗: {filename}")
        return

    # 画像変換
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img_hsv)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # --- 描画設定 (フォントサイズの一括指定) ---
    # ここで全体のサイズをコントロールします
    plt.rcParams.update({
        'font.size': 14,          # 全体の基本フォントサイズ
        'axes.titlesize': 18,     # 各グラフのタイトル (set_title)
        'axes.labelsize': 16,     # 軸ラベル (xlabel, ylabel)
        'xtick.labelsize': 12,    # X軸の目盛り数字
        'ytick.labelsize': 12,    # Y軸の目盛り数字
        'figure.titlesize': 24    # 全体のタイトル (suptitle)
    })

    # --- 描画設定 ---
    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle(f"{filename}")

    # 1. 元画像
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("1. Original Image")
    ax1.axis('off')

    # 2. RGBヒストグラム (正規化: Max=1.0)
    ax2 = fig.add_subplot(2, 2, 2)
    color_codes = ('b', 'g', 'r')
    labels = ('Blue', 'Green', 'Red')
    
    hists = []
    for i in range(3):
        hist = cv.calcHist([img], [i], None, [256], [0, 256])
        hists.append(hist)
    
    global_max = max([h.max() for h in hists])
    if global_max == 0: global_max = 1.0

    for i, col in enumerate(color_codes):
        normalized_hist = hists[i] / global_max
        ax2.plot(normalized_hist, color=col, label=labels[i], linewidth=1.5)
    
    ax2.set_title("2. RGB Histogram")
    ax2.set_xlabel("Pixel Value")
    ax2.set_ylabel("Relative Frequency (Max=1.0)")
    ax2.set_xlim([0, 255])
    ax2.set_ylim([0.0, 1.0])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # 3. HSV特性 (S-V分布)
    ax3 = fig.add_subplot(2, 2, 3)
    h2d = ax3.hist2d(s.flatten(), v.flatten(), bins=100, range=[[0, 255], [0, 255]], cmap='inferno', norm='log')
    ax3.set_title("3. S-V Distribution")
    ax3.set_xlabel("Saturation (0-255)")
    ax3.set_ylabel("Value (0-255)")
    fig.colorbar(h2d[3], ax=ax3, label="Pixel Count (Log)")

    # 4. 勾配プロット (Gradient Magnitude)
    ax4 = fig.add_subplot(2, 2, 4)
    
    # SobelフィルタでX方向、Y方向の微分を計算 (CV_64Fで計算精度を確保)
    sobelx = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    grad_display = cv.normalize(gradient_magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    # 勾配画像単体の保存
    grad_filename = name_without_ext + "_gradient_map.png"
    grad_save_path = os.path.join(map_output_dir, grad_filename)
    cv.imwrite(grad_save_path, grad_display)

    ax4.imshow(grad_display, cmap='gray')
    ax4.set_title("4. Gradient Magnitude")
    ax4.axis('off')

    # 解析グラフの保存
    save_name = name_without_ext + "_analyzed_gradient.png"
    save_path = os.path.join(graph_output_dir, save_name)
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

def process_folder(target_folder, base_output_folder):
    os.makedirs(base_output_folder, exist_ok=True)
    
    graph_dir = os.path.join(base_output_folder, "analysis_graphs")
    map_dir = os.path.join(base_output_folder, "gradient_maps")
    
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    
    print(f"出力先: {base_output_folder}")
    print(f"  -> グラフ保存先: {graph_dir}")
    print(f"  -> 勾配図保存先: {map_dir}")

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(target_folder, ext)))
        image_files.extend(glob(os.path.join(target_folder, ext.upper())))
    
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print("指定入力フォルダ内に画像ファイルが見つかりませんでした。")
        return

    print(f"対象ファイル数: {len(image_files)}")
    print("-" * 30)

    for img_path in image_files:
        analyze_and_save(img_path, graph_dir, map_dir)
    
    print("-" * 30)
    print("全ての処理が完了しました。")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"使用法: python {os.path.basename(sys.argv[0])} <入力画像フォルダ> <出力保存フォルダ>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(input_folder):
        print(f"エラー: 入力フォルダが見つかりません -> {input_folder}")
        sys.exit(1)
    
    if not os.path.isdir(input_folder):
        print(f"エラー: 入力パスはフォルダではありません -> {input_folder}")
        sys.exit(1)

    process_folder(input_folder, output_folder)
"""
指定された3つのフォルダ（'only_8bit', 'clahe_after_8bit', 'clahe_before_8bit'）
内の対応する画像セットを読み込み、
「左側に画像、右側にその画像のRGB輝度ヒストグラム」
という形式で縦に3つ並べた比較画像として保存するスクリプト。

前提とするフォルダ構造:
    <指定フォルダ>/
    ├── only_8bit/ (キャプション1)
    │   ├── image01.png
    │   └── ...
    ├── clahe_after_8bit/ (キャプション2)
    │   ├── image01_clahe.png
    │   └── ...
    └── clahe_before_8bit/ (キャプション3)
        ├── image01_clahe.png
        └── ...

実行方法:
    python plot_histogram_rgb_comparison.py <前提フォルダ構造の親フォルダパス>
"""

import glob
import os
import sys

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 'Agg'バックエンドを使用 (GUIなし環境でも動作)
matplotlib.use('Agg')

# --- 日本語キャプションのための設定 ---
try:
    import japanize_matplotlib
    japanize_matplotlib.japanize()
    print("[INFO] japanize_matplotlib を有効化しました。")
except ImportError:
    print("[警告] japanize_matplotlib が見つかりません。")
    print("      日本語キャプションが文字化けする場合があります。")
    print("      (インストール: pip install japanize-matplotlib)")


# --- Matplotlibのフォントサイズ設定 (ベーススクリプトから流用) ---
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# ----------- 設定値 (ベーススクリプトから流用) -----------
Y_AXIS_MAX_PERCENT = 20.0

# ----------- スクリプト固有の設定 -----------
FOLDER_CONFIG = {
    "only_8bit": "16bit→8bitへの変換のみ",
    "clahe_after_8bit": "16bit→8bitへの変換後、CLAHEによる鮮明化",
    "clahe_before_8bit": "CLAHEによる鮮明化後、16bit→8bitへの変換"
}
# 基準とするフォルダ (このフォルダにある画像名を元に処理)
BASE_FOLDER_NAME = list(FOLDER_CONFIG.keys())[0] # "only_8bit"
# 出力フォルダ名
OUTPUT_FOLDER_NAME = "HISTOGRAM_RGB_COMPARISON"
# 基準フォルダ以外に追加するサフィックス
SUFFIX_FOR_OTHERS = "_clahe"


def load_image_bgr_and_rgb(image_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    画像をBGR形式で読み込み、BGR配列とRGB配列(表示用)の両方を返す。
    """
    image_bgr = cv.imread(image_path, cv.IMREAD_COLOR)
    if image_bgr is None:
        # [警告] ではなく [INFO] に変更 (ファイルが見つからないのはエラーではないため)
        print(f"[INFO] 画像を読み込めませんでした (スキップ): {image_path}")
        return None, None

    if image_bgr.dtype != np.uint8:
        print(f"[警告] 8bit画像(uint8)ではありません ({image_bgr.dtype}): {os.path.basename(image_path)}")
        return None, None

    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

    return image_bgr, image_rgb


def plot_rgb_histogram(ax, image_bgr: np.ndarray, title: str):
    """
    指定されたMatplotlibのAxesに、RGBヒストグラムを描画する。
    """
    channels = cv.split(image_bgr)
    colors = ("blue", "green", "red")
    labels = ("B", "G", "R")

    total_pixels_per_channel = image_bgr.shape[0] * image_bgr.shape[1]
    weights = np.ones(total_pixels_per_channel) / total_pixels_per_channel * 100

    bins = 256
    max_val_exclusive = 256

    for channel, color, label in zip(channels, colors, labels):
        ax.hist(
            channel.ravel(),
            bins=bins,
            range=[0, max_val_exclusive],
            color=color,
            edgecolor=color,
            alpha=0.4,
            weights=weights,
            label=label
        )
    
    ax.set_title(title)
    ax.set_xlabel("RGB Value (0-255)")
    ax.set_ylabel("Percentage of Pixels (%)")
    ax.set_xlim([0, 256])

    if Y_AXIS_MAX_PERCENT is not None:
        ax.set_ylim(0, Y_AXIS_MAX_PERCENT)

    ax.legend()


def plot_comparison_with_images(
    image_paths: list[str], 
    captions: list[str], 
    output_path: str
):
    """
    3セットの「画像とRGBヒストグラム」を1つの画像にまとめて保存する。
    """
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 20))

    images_data = []
    
    # --- 1. 画像の読み込み ---
    all_loaded = True
    for path in image_paths:
        bgr, rgb = load_image_bgr_and_rgb(path)
        if bgr is None:
            all_loaded = False
            break
        images_data.append({'bgr': bgr, 'rgb': rgb})

    # 1つでも読み込めなかったらグラフ生成を中止
    if not all_loaded:
        print(f"[警告] 3つの画像セットが揃いませんでした。 {output_path} の生成をスキップします。")
        plt.close(fig)
        return

    # --- 2. 各行にプロット ---
    for i in range(3):
        ax_img = axes[i, 0]
        ax_hist = axes[i, 1]
        data = images_data[i]
        
        ax_img.imshow(data['rgb'])
        ax_img.axis('off') 
        
        plot_rgb_histogram(ax_hist, data['bgr'], captions[i])

    plt.tight_layout()

    # --- 3. プロットをファイルに保存 ---
    try:
        plt.savefig(output_path)
        print(f"[OK] 比較グラフを保存しました: {output_path}")
    except Exception as e:
        print(f"[エラー] ファイルの保存中にエラーが発生しました: {e}")
    finally:
        plt.close(fig)


def process_comparison(base_results_folder: str):
    """
    指定された親フォルダ内のサブフォルダを基準に、
    対応する画像のRGBヒストグラムを比較・保存する。
    """
    if not os.path.isdir(base_results_folder):
        print(f"[エラー] 指定されたフォルダが存在しません: {base_results_folder}")
        return

    folder_names = list(FOLDER_CONFIG.keys())
    captions = list(FOLDER_CONFIG.values())
    
    sub_dirs = [os.path.join(base_results_folder, name) for name in folder_names]

    all_dirs_exist = True
    for dir_path, name in zip(sub_dirs, folder_names):
        if not os.path.isdir(dir_path):
            print(f"[エラー] サブフォルダ '{name}' が存在しません: {dir_path}")
            all_dirs_exist = False
    
    if not all_dirs_exist:
        print("[エラー] 必要なサブフォルダが揃っていません。処理を中断します。")
        return

    base_results_folder = os.path.normpath(base_results_folder)
    main_output_folder = os.path.join(base_results_folder, OUTPUT_FOLDER_NAME)
    
    os.makedirs(main_output_folder, exist_ok=True)
    print(f"[INFO] メイン出力フォルダ: {main_output_folder}")

    # 基準フォルダ(リストの先頭)内のPNG画像を基準に処理
    base_dir_path = sub_dirs[0]
    image_files = sorted(glob.glob(os.path.join(base_dir_path, "*.png")))

    if not image_files:
        print(f"[INFO] '{BASE_FOLDER_NAME}' フォルダに処理対象の画像がありません。")
        return
        
    print(f"\n--- 処理開始 ({len(image_files)} 件) ---")
    
    for base_img_path in image_files:
        base_filename = os.path.basename(base_img_path)
        
        # --- ★★★ 変更点 ★★★ ---
        # サフィックス付きのファイル名を生成
        name, ext = os.path.splitext(base_filename)
        clahe_filename = f"{name}{SUFFIX_FOR_OTHERS}{ext}"

        # 3つの画像のパスをリスト化
        image_paths = [
            # 1. 基準フォルダ (only_8bit) -> サフィックスなし
            base_img_path, 
            
            # 2. clahe_after_8bit -> サフィックスあり
            os.path.join(sub_dirs[1], clahe_filename),
            
            # 3. clahe_before_8bit -> サフィックスあり
            os.path.join(sub_dirs[2], clahe_filename)
        ]

        # 出力パスを生成 (基準ファイル名からサフィックスを除いた名前を使用)
        output_path = os.path.join(main_output_folder, f"{name}_hist_rgb_comparison.png")

        # 3つのファイルがすべて揃っているか確認 (このチェックは plot_comparison_with_images 内に移動)
        plot_comparison_with_images(image_paths, captions, output_path)
        # --- ★★★ 変更ここまで ★★★ ---


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"使い方: python {os.path.basename(__file__)} <結果フォルダの親パス>")
        print(f"  (例: 内部に '{BASE_FOLDER_NAME}' などがあるフォルダ)")
        sys.exit(1)
        
    root_folder_path = sys.argv[1]
    process_comparison(root_folder_path)
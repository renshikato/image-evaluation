"""
指定フォルダ内の16bit画像（TIFまたはPNG）を読み込み、
2つの異なる方法で8bit画像（PNG）に変換して保存するスクリプト。

同時に、入力画像1枚ごとに、
- 元画像 (16bit)
- 処理後画像1 (Normalize)
- 処理後画像2 (Normalize_Clipped)
の計3枚の画像と、それら3つの輝度ヒストグラムを比較した
グラフを、1枚のPNG画像として保存する。

処理方法:
1. Normalize (単純な正規化)
2. Normalize_Clipped (上下1%のパーセンタイルでクリップした正規化)

実行方法:
    python3 process_16to8_conversion_for_comparison.py <入力フォルダパス> <出力フォルダパス>
"""

import glob
import os
import sys

import cv2 as cv
import numpy as np

# Matplotlibをインポート (ヒストグラムプロット用)
import matplotlib
matplotlib.use('Agg')  # GUIなし環境用
import matplotlib.pyplot as plt

# --- Matplotlibのフォントサイズ設定 (ヒストグラム比較用) ---
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# --- ユーザー設定パラメータ (クリッピング率) ---
CLIP_LOW_PERCENT = 3.0
CLIP_HIGH_PERCENT = 97.0

def convert_16bit_to_8bit_normalize(img_16bit: np.ndarray) -> np.ndarray:
    """
    方法1: 正規化 (NORM_MINMAX)
    """
    img_8bit = cv.normalize(
        img_16bit, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U
    )
    return img_8bit

def convert_16bit_to_8bit_clipped(
    img_16bit: np.ndarray,
    low_percent: float,
    high_percent: float
) -> np.ndarray:
    """
    方法2: パーセンタイルベースのクリッピングを用いた正規化。
    この関数は low_percent と high_percent を必須の引数として要求する。
    """
    vmin = np.percentile(img_16bit, low_percent)
    vmax = np.percentile(img_16bit, high_percent)

    if vmin >= vmax:
        return np.zeros_like(img_16bit, dtype=np.uint8)

    img_clipped = np.clip(img_16bit, vmin, vmax)
    img_8bit = cv.normalize(
        img_clipped, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U
    )
    return img_8bit

def get_gray_image(image_raw: np.ndarray) -> np.ndarray | None:
    """
    ヒストグラム計算用に、画像を8bitまたは16bitのグレースケールに変換する。
    
    Args:
        image_raw: 入力画像 (カラーまたはグレー, uint8またはuint16)
    
    Returns:
        グレースケール画像。サポート外のdtypeの場合はNone。
    """
    if len(image_raw.shape) == 3 and image_raw.shape[2] >= 3:
        gray_image = cv.cvtColor(image_raw, cv.COLOR_BGR2GRAY)
    else:
        gray_image = image_raw
    
    if gray_image.dtype not in [np.uint8, np.uint16]:
        print(f"[警告] サポート外のdtypeです ({gray_image.dtype})")
        return None
    return gray_image

def get_display_image_and_cmap(image_raw: np.ndarray) -> (np.ndarray, str | None):
    """
    Matplotlib (imshow) で表示するための画像とカラーマップを取得する。
    
    - 16bit画像は、上位8bitを抽出 (>> 8) して8bit画像に変換する。
    - カラー画像はBGRからRGBに変換する。
    - グレースケール画像の場合は cmap='gray' を返す。
    
    Args:
        image_raw: 入力画像 (uint8またはuint16)
    
    Returns:
        (表示用8bit画像, カラーマップ(strまたはNone))
    """
    cmap = None
    if len(image_raw.shape) == 3 and image_raw.shape[2] >= 3:
        image_display = cv.cvtColor(image_raw, cv.COLOR_BGR2RGB)
    else:
        image_display = image_raw
        cmap = 'gray'

    if image_display.dtype == np.uint16:
        image_display_converted = (image_display >> 8).astype(np.uint8)
        return image_display_converted, cmap
    else:
        return image_display, cmap

def plot_single_hist_overlay(ax, gray_image: np.ndarray, label: str, color: str, alpha: float = 0.4):
    """
    指定されたMatplotlibのAxesに、単一のヒストグラムを割合(0.0-1.0)で重ね描きする。
    
    Args:
        ax: 描画対象の Matplotlib Axes
        gray_image: ヒストグラムを計算するグレースケール画像 (uint8またはuint16)
        label: 凡例用のラベル
        color: プロットの色
        alpha: 透明度
    """
    if gray_image.dtype == np.uint8:
        max_val_exclusive = 256
        bins = 256
    elif gray_image.dtype == np.uint16:
        max_val_exclusive = 65536
        bins = 256
    else:
        return

    weights = np.ones_like(gray_image.ravel()) / gray_image.size
    
    ax.hist(
        gray_image.ravel(),
        bins=bins,
        range=[0, max_val_exclusive],
        color=color,
        edgecolor=color,
        alpha=alpha,
        weights=weights,
        label=label
    )

def plot_comparison_figure(
    img_16bit: np.ndarray,
    img_norm: np.ndarray,
    img_clipped: np.ndarray,
    output_path: str,
    base_name: str,
    low_percent: float,   # クリッピング率を引数で受け取る
    high_percent: float  # クリッピング率を引数で受け取る
):
    """
    3枚の画像と、3つのヒストグラム比較グラフを1枚の画像に出力する。
    
    - 左側に画像3枚 (Original, Normalize, Clipped) を表示。
    - 右側にヒストグラム3種 (16bit, 8bit-Norm, 8bit-Clipped) を重ねて表示。
    """
    # --- 1. ヒストグラム計算用のグレースケール画像を取得 ---
    gray_16bit = get_gray_image(img_16bit)
    gray_norm = get_gray_image(img_norm)
    gray_clipped = get_gray_image(img_clipped)

    if gray_16bit is None or gray_norm is None or gray_clipped is None:
        print(f"[エラー] ヒストグラム生成のための画像取得に失敗: {base_name}")
        return

    # --- 2. 表示用の8bit画像を取得 ---
    disp_16bit, cmap_16bit = get_display_image_and_cmap(img_16bit)
    disp_norm, cmap_norm = get_display_image_and_cmap(img_norm)
    disp_clipped, cmap_clipped = get_display_image_and_cmap(img_clipped)

    # --- 3. プロットのレイアウト準備 (3行2列) ---
    fig = plt.figure(figsize=(18, 15))
    fig.suptitle(f"Image and Histogram Comparison\n({base_name})", fontsize=24)

    # --- 3a. 左側: 画像表示 (3行1列目) ---
    ax_img1 = plt.subplot(3, 2, 1)
    ax_img1.set_title("Original (16bit)")
    ax_img1.imshow(disp_16bit, cmap=cmap_16bit)
    ax_img1.axis('off')

    ax_img2 = plt.subplot(3, 2, 3)
    ax_img2.set_title("Normalize (8bit)")
    ax_img2.imshow(disp_norm, cmap=cmap_norm)
    ax_img2.axis('off')

    ax_img3 = plt.subplot(3, 2, 5)
    ax_img3.set_title(f"Normalize_Clipped ({low_percent}%-{high_percent}%)")
    ax_img3.imshow(disp_clipped, cmap=cmap_clipped)
    ax_img3.axis('off')

    # --- 3b. 右側: ヒストグラム表示 (3行分を結合) ---
    ax_hist1 = plt.subplot(3, 2, (2, 6))
    ax_hist2 = ax_hist1.twiny()

    # --- Clipped の凡例 (label) を動的に生成 ---
    clipped_label = f"Clipped ({low_percent}%-{high_percent}%)"
    
    # 8bitヒストグラムをプロット
    plot_single_hist_overlay(ax_hist1, gray_norm, "Normalize (8bit)", "blue")
    plot_single_hist_overlay(ax_hist1, gray_clipped, clipped_label, "red")

    # 16bitヒストグラムをプロット
    plot_single_hist_overlay(ax_hist2, gray_16bit, "Original (16bit)", "black", alpha=0.7)

    # 軸の設定
    ax_hist1.set_xlabel("Luminance (8bit: 0-255)", color="blue")
    ax_hist1.set_xlim(0, 256)
    ax_hist1.tick_params(axis='x', labelcolor="blue")
    ax_hist2.set_xlabel("Luminance (16bit: 0-65535)", color="black")
    ax_hist2.set_xlim(0, 65536)
    ax_hist2.tick_params(axis='x', labelcolor="black")
    
    # Y軸ラベルを "Proportion" (割合) にする
    ax_hist1.set_ylabel("Proportion of Pixels (0.0-1.0)")
    
    # Y軸の範囲を 0.0 から 1.0 に固定
    ax_hist1.set_ylim(0, 1.0)
    
    # 凡例 (Legend) をまとめる
    lines1, labels1 = ax_hist1.get_legend_handles_labels()
    lines2, labels2 = ax_hist2.get_legend_handles_labels()
    ax_hist2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # --- 4. プロットをファイルに保存 ---
    try:
        plt.savefig(output_path)
        print(f"  -> 比較グラフを保存: {output_path}")
    except Exception as e:
        print(f"[エラー] 比較グラフ画像の保存中にエラーが発生しました: {e}")
    finally:
        plt.close(fig)


def process_folder(input_dir: str, output_dir: str):
    """
    指定されたフォルダ内の16bit画像を処理し、2つの方法で8bit変換し、
    ヒストグラム比較画像を出力する。
    """
    if not os.path.isdir(input_dir):
        print(f"[ERROR] 入力フォルダが存在しません: {input_dir}")
        return
    
    # グローバル変数の設定値から、フォルダ名セーフな文字列を生成
    low_str = str(CLIP_LOW_PERCENT).replace('.', 'p')
    high_str = str(CLIP_HIGH_PERCENT).replace('.', 'p')
    
    # フォルダ名にクリッピング率を含める
    clip_suffix = f"L{low_str}_H{high_str}"
    
    # 出力先のサブフォルダを作成
    output_normalize_dir = os.path.join(output_dir, "normalize")
    output_clipped_dir = os.path.join(output_dir, f"normalize_clipped_{clip_suffix}")
    output_hist_dir = os.path.join(output_dir, f"histogram_comparison_{clip_suffix}")

    os.makedirs(output_normalize_dir, exist_ok=True)
    os.makedirs(output_clipped_dir, exist_ok=True)
    os.makedirs(output_hist_dir, exist_ok=True)

    patterns = [
        "*.tif", "*.TIF", "*.tiff", "*.TIFF",
        "*.png", "*.PNG"
    ]
    image_files = []
    for pattern in patterns:
        image_files.extend(
            glob.glob(os.path.join(input_dir, pattern))
        )
    
    image_files.sort()

    if not image_files:
        print(f"[INFO] 入力フォルダ内に処理対象の画像が見つかりません: {input_dir}")
        return

    print(f"[INFO] {len(image_files)} 件の画像を処理します...")
    # 実行する設定値をログに出力
    print(f"[INFO] クリッピング率: Low={CLIP_LOW_PERCENT}%, High={CLIP_HIGH_PERCENT}%")
    print(f"[INFO] クリップ画像出力先: {output_clipped_dir}")
    print(f"[INFO] グラフ出力先: {output_hist_dir}")
    
    for file_path in image_files:
        base_name = os.path.basename(file_path)
        name, _ = os.path.splitext(base_name)
        output_filename = f"{name}.png"

        img_16bit = cv.imread(file_path, cv.IMREAD_UNCHANGED)
        # (None チェック, dtype チェック ...)
        if img_16bit is None:
            print(f"[WARNING] 読み込み失敗: {base_name}")
            continue
        if img_16bit.dtype != np.uint16:
            print(f"[WARNING] 16bit画像(uint16)ではありません: {base_name} (dtype: {img_16bit.dtype})")
            continue

        print(f"--- 処理中: {base_name} (16bit Min: {img_16bit.min()}, Max: {img_16bit.max()}) ---")

        try:
            # 方法1: 正規化
            img_8bit_normalize = convert_16bit_to_8bit_normalize(img_16bit)
            save_path = os.path.join(output_normalize_dir, output_filename)
            cv.imwrite(save_path, img_8bit_normalize)
            print(f"  -> Normalize 保存完了")
            
            # --- 変数を使用 ---
            # 方法2: クリッピング正規化
            img_8bit_clipped = convert_16bit_to_8bit_clipped(
                img_16bit, CLIP_LOW_PERCENT, CLIP_HIGH_PERCENT
            )
            save_path = os.path.join(output_clipped_dir, output_filename)
            cv.imwrite(save_path, img_8bit_clipped)
            print(f"  -> Clipped ({CLIP_LOW_PERCENT}%-{CLIP_HIGH_PERCENT}%) 保存完了") # ログにも表示

            # --- ヒストグラム比較プロットの呼び出し ---
            hist_output_path = os.path.join(output_hist_dir, f"{name}_hist_comparison.png")
            plot_comparison_figure(
                img_16bit,
                img_8bit_normalize,
                img_8bit_clipped,
                hist_output_path,
                base_name,
                CLIP_LOW_PERCENT,  # グローバル変数を渡す
                CLIP_HIGH_PERCENT  # グローバル変数を渡す
            )

        except Exception as e:
            print(f"[ERROR] 変換または保存中にエラーが発生しました ({base_name}): {e}")

    print(f"\n[INFO] 全ての処理が完了しました。出力先: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        script_name = os.path.basename(sys.argv[0])
        print(f"使用方法: python {script_name} <入力フォルダパス> <出力フォルダパス>")
        print(f"例: python {script_name} ./my_16bit_images ./my_8bit_results")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    process_folder(input_folder, output_folder)
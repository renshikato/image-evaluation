"""
指定フォルダ内の16bit画像（TIF/PNG）を読み込み、
1. グレーワールド法によるホワイトバランス調整（16bit）
2. パーセンタイルクリッピング正規化による8bit変換
の順序で処理を実行するスクリプト。

1. 処理後の8bit画像を 'wb_normalized_...' フォルダに保存する。
2. 処理前後のRGBヒストグラム比較グラフを 'rgb_histogram_...' フォルダに保存する。
3. 処理前後の「輝度」ヒストグラム比較グラフを 'luminance_histogram_...' フォルダに保存する。
"""

import glob
import os
import sys

import cv2 as cv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Matplotlibのフォントサイズ設定 ---
plt.rcParams.update({
    'font.size': 20,        # 全体の基準サイズ
    'figure.titlesize': 32, # 画像全体のメインタイトル (fig.suptitle)
    'axes.titlesize': 22,   # 各グラフのタイトル (ax.set_title)
    'axes.labelsize': 20,   # X軸・Y軸のラベル名 (ax.set_xlabel/ylabel)
    'xtick.labelsize': 20,  # X軸の目盛り数値
    'ytick.labelsize': 20,  # Y軸の目盛り数値
    'legend.fontsize': 20,  # 凡例（レジェンド）
})
Y_AXIS_MAX_RATIO = 1.0

# --- ユーザー設定パラメータ (クリッピング率) ---
CLIP_LOW_PERCENT = 2.0
CLIP_HIGH_PERCENT = 98.0

def apply_white_balance_gray_world(img_input: np.ndarray) -> np.ndarray:
    """
    グレーワールド法でホワイトバランスを調整 (8bitまたは16bit対応)
    """
    if len(img_input.shape) < 3 or img_input.shape[2] < 3:
        print("[INFO] グレースケール画像のため、ホワイトバランスをスキップします。")
        return img_input
    img_float = img_input[:, :, :3].astype(np.float32)
    b_avg = np.mean(img_float[:, :, 0])
    g_avg = np.mean(img_float[:, :, 1])
    r_avg = np.mean(img_float[:, :, 2])
    gray_avg = (b_avg + g_avg + r_avg) / 3.0
    b_scale = gray_avg / b_avg if b_avg > 0 else 1.0
    g_scale = gray_avg / g_avg if g_avg > 0 else 1.0
    r_scale = gray_avg / r_avg if r_avg > 0 else 1.0
    img_float[:, :, 0] *= b_scale
    img_float[:, :, 1] *= g_scale
    img_float[:, :, 2] *= r_scale
    if img_input.dtype == np.uint8:
        output_max_val = 255
    else: 
        output_max_val = 65535
    img_wb_float = np.clip(img_float, 0, output_max_val)
    output_img = img_input.copy()
    output_img[:, :, :3] = img_wb_float.astype(img_input.dtype)
    return output_img


def convert_16bit_to_8bit_clipped(
    img_16bit: np.ndarray,
    low_percent: float,
    high_percent: float
) -> np.ndarray:
    """
    パーセンタイルベースのクリッピングを用いた正規化 (Normalize_Clipped)
    (16bit入力 -> 8bit出力)
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
    """
    if len(image_raw.shape) == 3 and image_raw.shape[2] >= 3:
        if image_raw.dtype == np.uint8:
            gray_image = cv.cvtColor(image_raw, cv.COLOR_BGR2GRAY)
        else:
            # uint16のBGR->GRAYはcvtColorがサポートしていない場合があるため、
            # 平均値でグレースケール化する (より正確には重み付け平均だが簡略化)
            # または、Gチャネルを輝度とみなす
            # ここではGチャネルを採用
            gray_image = image_raw[:, :, 1] 
            # もしGチャネルが不適切なら cv.cvtColor(image_raw.astype(np.float32), cv.COLOR_BGR2GRAY).astype(np.uint16)
    else:
        gray_image = image_raw
    
    if gray_image.dtype not in [np.uint8, np.uint16]:
        print(f"[警告] サポート外のdtypeです ({gray_image.dtype})")
        return None
    return gray_image


def get_display_image_for_plot(image_raw: np.ndarray) -> np.ndarray:
    """
    plt.imshowで表示するための8bit RGB画像を返す。
    BGR -> RGB, 16bit -> 8bit (ビットシフト)
    """
    if len(image_raw.shape) == 3 and image_raw.shape[2] >= 3:
        image_display = cv.cvtColor(image_raw[:, :, :3], cv.COLOR_BGR2RGB)
    else:
        image_display = image_raw
        
    if image_display.dtype == np.uint16:
        image_display_converted = (image_display >> 8).astype(np.uint8)
        return image_display_converted
    else:
        return image_display


def plot_rgb_hist(ax, img_bgr: np.ndarray):
    """
    指定されたAxesに、単一の画像のRGBヒストグラムを描画する。
    「最大値」が1.0になるように正規化する。
    """
    if img_bgr.dtype == np.uint8:
        max_val_exclusive = 256
        xlabel = "RGB Value (8bit: 0-255)"
    elif img_bgr.dtype == np.uint16:
        max_val_exclusive = 65536
        xlabel = "RGB Value (16bit: 0-65535)"
    else:
        print(f"[WARN] サポート外のdtype ({img_bgr.dtype})")
        return

    bins = 256 
    channels = cv.split(img_bgr[:, :, :3])
    colors = ('blue', 'green', 'red')
    labels = ('B Channel', 'G Channel', 'R Channel')
    
    # 1.先にヒストグラムを計算し、全チャネル中の最大カウント数を探す
    counts = []
    for (channel) in channels:
        c, _ = np.histogram(
            channel.ravel(), bins=bins, range=[0, max_val_exclusive]
        )
        counts.append(c)
    
    # (np.max が空の配列にエラーを出すのを防ぐ)
    if not counts or all(c.size == 0 for c in counts):
        print("[WARN] ヒストグラムデータが空です。")
        max_count = 1.0
    else:
        max_count = max(np.max(c) for c in counts)
        if max_count == 0:
            max_count = 1.0 # 全て0の場合のゼロ除算防止
    
    # 2. weight を「最大カウント数」で割るように変更
    for (channel, color, label) in zip(channels, colors, labels):
        # 各ピクセルの重みを　1 / max_count にする
        weights = np.ones_like(channel.ravel()) / max_count
        
        ax.hist(
            channel.ravel(), bins=bins, range=[0, max_val_exclusive],
            color=color, edgecolor=color, alpha=0.7,
            weights=weights, label=label
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized Count (Max=1.0)")
    ax.set_xlim(0, max_val_exclusive)
    if Y_AXIS_MAX_RATIO is not None:
        ax.set_ylim(0, Y_AXIS_MAX_RATIO)
    ax.legend(loc='upper right')


def plot_luminance_hist(ax, img_color_or_gray: np.ndarray):
    """
    指定されたAxesに、単一の画像の「輝度」ヒストグラムを描画する。
    「最大値」が1.0になるように正規化する。
    """
    gray_image = get_gray_image(img_color_or_gray)
    if gray_image is None:
        return

    if gray_image.dtype == np.uint8:
        max_val_exclusive = 256
        xlabel = "Luminance (8bit: 0-255)"
    elif gray_image.dtype == np.uint16:
        max_val_exclusive = 65536
        xlabel = "Luminance (16bit: 0-65535)"
    else:
        return

    bins = 256 
    
    # 1. 先にヒストグラムを計算して最大頻度（max_count）を取得
    counts, _ = np.histogram(
        gray_image.ravel(), bins=bins, range=[0, max_val_exclusive]
    )
    
    if counts.size == 0:
        max_count = 1.0
    else:
        max_count = counts.max()
    
    if max_count == 0:
        max_count = 1.0

    # 2. 重みを 1 / max_count に設定
    weights = np.ones_like(gray_image.ravel()) / max_count
        
    ax.hist(
        gray_image.ravel(),
        bins=bins,
        range=[0, max_val_exclusive],
        color='black',
        edgecolor='black',
        alpha=0.7,
        weights=weights
    )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized Count (Max=1.0)")
    ax.set_xlim(0, max_val_exclusive)
    if Y_AXIS_MAX_RATIO is not None:
        ax.set_ylim(0, Y_AXIS_MAX_RATIO)


def plot_rgb_comparison_figure(
    img_before: np.ndarray,
    img_after: np.ndarray,
    output_path: str, 
    base_name: str,
    clip_percents: str
):
    """
    処理前後の画像と、その「RGB」ヒストグラム比較グラフを出力する。
    """
    disp_before = get_display_image_for_plot(img_before)
    disp_after = get_display_image_for_plot(img_after)

    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f"RGB Comparison\n({base_name})")

    # 左上: 元画像
    ax_img1 = plt.subplot(2, 2, 1)
    ax_img1.set_title(f"Original\n (16bit / Displayed by Bit Shift)")
    ax_img1.imshow(disp_before)
    ax_img1.axis('off')

    # 右上: 元画像のRGBヒストグラム (16bit)
    ax_hist1 = plt.subplot(2, 2, 2)
    ax_hist1.set_title("Original RGB Histogram (16bit)")
    plot_rgb_hist(ax_hist1, img_before) # 変更

    # 左下: 処理後画像
    ax_img2 = plt.subplot(2, 2, 3)
    ax_img2.set_title(f"WB + Normalize_Clipped\n (8bit, {clip_percents}%)")
    ax_img2.imshow(disp_after)
    ax_img2.axis('off')

    # 右下: 処理後画像のRGBヒストグラム (8bit)
    ax_hist2 = plt.subplot(2, 2, 4)
    ax_hist2.set_title("(WB + Normalized) RGB Histogram (8bit)")
    plot_rgb_hist(ax_hist2, img_after) # 変更

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    try:
        plt.savefig(output_path)
        print(f"  -> RGB比較グラフを保存: {output_path}")
    except Exception as e:
        print(f"[エラー] RGB比較グラフ画像の保存中にエラーが発生しました: {e}")
    finally:
        plt.close(fig)


def plot_luminance_comparison_figure(
    img_before: np.ndarray,
    img_after: np.ndarray,
    output_path: str, 
    base_name: str,
    clip_percents: str
):
    """
    処理前後の画像と、その「輝度」ヒストグラム比較グラフを出力する。
    """
    disp_before = get_display_image_for_plot(img_before)
    disp_after = get_display_image_for_plot(img_after)

    fig = plt.figure(figsize=(24, 12))
    fig.suptitle(f"Luminance Comparison\n({base_name})")

    # 左上: 元画像
    ax_img1 = plt.subplot(2, 2, 1)
    ax_img1.set_title(f"Original\n (16bit / Displayed by Bit Shift)")
    ax_img1.imshow(disp_before)
    ax_img1.axis('off')

    # 右上: 元画像の輝度ヒストグラム (16bit)
    ax_hist1 = plt.subplot(2, 2, 2)
    ax_hist1.set_title("Original Luminance Histogram (16bit)")
    plot_luminance_hist(ax_hist1, img_before) # 輝度ヒストグラム

    # 左下: 処理後画像
    ax_img2 = plt.subplot(2, 2, 3)
    ax_img2.set_title(f"WB + Normalize_Clipped\n (8bit, {clip_percents}%)")
    ax_img2.imshow(disp_after)
    ax_img2.axis('off')

    # 右下: 処理後画像の輝度ヒストグラム (8bit)
    ax_hist2 = plt.subplot(2, 2, 4)
    ax_hist2.set_title("(WB + Normalized) Luminance Histogram (8bit)")
    plot_luminance_hist(ax_hist2, img_after) # 輝度ヒストグラム

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    try:
        plt.savefig(output_path)
        print(f"  -> 輝度比較グラフを保存: {output_path}")
    except Exception as e:
        print(f"[エラー] 輝度比較グラフ画像の保存中にエラーが発生しました: {e}")
    finally:
        plt.close(fig)


def process_folder(input_dir: str, output_parent_dir: str):
    """
    指定されたフォルダ内の16bit画像に対し、
    1. ホワイトバランス
    2. クリッピング正規化 (8bit化)
    の順で処理を実行し、結果とグラフを保存する。
    """
    if not os.path.isdir(input_dir):
        print(f"[ERROR] 入力フォルダが存在しません: {input_dir}")
        return

    # グローバル変数からフォルダ名サフィックスを生成
    low_str = str(CLIP_LOW_PERCENT).replace('.', 'p')
    high_str = str(CLIP_HIGH_PERCENT).replace('.', 'p')
    clip_suffix = f"L{low_str}_H{high_str}"
    
    # --- 3つの出力フォルダ ---
    output_dir_img = os.path.join(output_parent_dir, f"wb_normalized_{clip_suffix}")
    output_dir_hist_rgb = os.path.join(output_parent_dir, f"rgb_histogram_comparison_{clip_suffix}")
    output_dir_hist_lum = os.path.join(output_parent_dir, f"luminance_histogram_comparison_{clip_suffix}")
    
    os.makedirs(output_dir_img, exist_ok=True)
    os.makedirs(output_dir_hist_rgb, exist_ok=True)
    os.makedirs(output_dir_hist_lum, exist_ok=True)
    print(f"[INFO] 処理画像 出力先: {output_dir_img}")
    print(f"[INFO] RGB比較グラフ 出力先: {output_dir_hist_rgb}")
    print(f"[INFO] 輝度比較グラフ 出力先: {output_dir_hist_lum}")

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
        print(f"[INFO] 入力フォルダ内に処理対象の16bit画像が見つかりません: {input_dir}")
        return

    print(f"\n[INFO] {len(image_files)} 件の画像を処理します...")
    print(f"[INFO] クリッピング率: Low={CLIP_LOW_PERCENT}%, High={CLIP_HIGH_PERCENT}%")
    
    clip_percents_str = f"{CLIP_LOW_PERCENT}-{CLIP_HIGH_PERCENT}"
    
    for file_path in image_files:
        base_name = os.path.basename(file_path)
        name, _ = os.path.splitext(base_name)
        
        output_img_filename = f"{name}_wb_norm.png"
        output_img_path = os.path.join(output_dir_img, output_img_filename)
        
        # --- 2種類のヒストグラムパス ---
        output_hist_rgb_filename = f"{name}_wb_norm_rgb_hist.png"
        output_hist_rgb_path = os.path.join(output_dir_hist_rgb, output_hist_rgb_filename)
        output_hist_lum_filename = f"{name}_wb_norm_lum_hist.png"
        output_hist_lum_path = os.path.join(output_dir_hist_lum, output_hist_lum_filename)

        try:
            img_16bit_orig = cv.imread(file_path, cv.IMREAD_UNCHANGED)
            
            if img_16bit_orig is None:
                print(f"[WARNING] 読み込み失敗 (スキップ): {base_name}")
                continue
            if img_16bit_orig.dtype != np.uint16:
                print(f"[WARNING] 16bit(uint16)ではありません (スキップ): {base_name}")
                continue

            print(f"--- 処理中: {base_name} (dtype: {img_16bit_orig.dtype}) ---")
            
            img_16bit_wb = apply_white_balance_gray_world(img_16bit_orig)
            
            img_8bit_final = convert_16bit_to_8bit_clipped(
                img_16bit_wb, CLIP_LOW_PERCENT, CLIP_HIGH_PERCENT
            )
            
            cv.imwrite(output_img_path, img_8bit_final)
            print(f"  -> 処理画像を保存: {output_img_path}")

            # 5. RGB比較プロット画像を保存
            plot_rgb_comparison_figure(
                img_16bit_orig,
                img_8bit_final,
                output_hist_rgb_path,
                base_name,
                clip_percents_str
            )

            # --- 輝度比較プロット画像を保存 ---
            plot_luminance_comparison_figure(
                img_16bit_orig,
                img_8bit_final,
                output_hist_lum_path,
                base_name,
                clip_percents_str
            )

        except Exception as e:
            print(f"[ERROR] 変換または保存中にエラーが発生しました ({base_name}): {e}")

    print(f"\n[INFO] 全ての処理が完了しました。")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        script_name = os.path.basename(sys.argv[0])
        print(f"使用方法: python3 {script_name} <入力フォルダパス> <出力親フォルダパス>")
        print(f"例: python3 {script_name} ./my_16bit_images ./my_wb_norm_results")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2] 

    process_folder(input_folder, output_folder)
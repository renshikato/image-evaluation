import sys
import os
import cv2 as cv
import numpy as np
from glob import glob

# ==========================================
# パラメータ設定 (画像の状態に合わせて調整してください)
# ==========================================
# 1. 除去したいマリンスノーの最大サイズ (奇数推奨: 15, 21, 31など)
#    値を大きくすると、大きなボケたゴミも消せますが、魚を誤検出するリスクが増えます。
KERNEL_SIZE = 31

# 2. 感度閾値 (0〜255)
#    値を小さくする(例:10) -> 薄いノイズも消えるが、背景も誤検出しやすい
#    値を大きくする(例:40) -> くっきりした白い点だけ消す
THRESHOLD = 20

# 3. インペイントの参照半径
#    通常は3〜5程度でOK
INPAINT_RADIUS = 3
# ==========================================

def remove_marine_snow(img_path, output_root):
    """
    画像を読み込み、モルフォロジー処理でノイズを除去して保存する
    """
    filename = os.path.basename(img_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # 画像読み込み
    img = cv.imread(img_path)
    if img is None:
        print(f"読み込み失敗: {filename}")
        return

    # --- 1. ノイズ検出 (モルフォロジー・トップハット変換) ---
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # 構造化要素（楕円形）を作成
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    
    # トップハット変換: (元画像 - オープニング)
    # これにより「背景より局所的に明るいもの」が抽出される
    tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    
    # 2値化してマスクを作成
    _, mask = cv.threshold(tophat, THRESHOLD, 255, cv.THRESH_BINARY)
    
    # マスクの微調整: ノイズの周辺も確実に消すために少し膨張させる
    mask = cv.dilate(mask, None, iterations=1)

    # --- 2. ノイズ除去 (インペイント) ---
    # マスク部分を周囲の色で修復
    cleaned_img = cv.inpaint(img, mask, INPAINT_RADIUS, cv.INPAINT_TELEA)

    # --- 3. 保存処理 ---
    # 出力先サブフォルダ
    clean_dir = os.path.join(output_root, "result_images")
    mask_dir = os.path.join(output_root, "debug_masks")
    
    # 除去後画像の保存
    save_path_clean = os.path.join(clean_dir, f"{name_without_ext}_cleaned.jpg")
    cv.imwrite(save_path_clean, cleaned_img)
    
    # マスク画像の保存 (調整確認用)
    save_path_mask = os.path.join(mask_dir, f"{name_without_ext}_mask.jpg")
    cv.imwrite(save_path_mask, mask)
    
    print(f"Processed: {filename}")

def process_folder(input_dir, output_dir):
    # 必要なフォルダの作成
    os.makedirs(os.path.join(output_dir, "result_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "debug_masks"), exist_ok=True)

    print(f"入力: {input_dir}")
    print(f"出力: {output_dir}")
    print("-" * 30)

    # 画像ファイルの検索
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(input_dir, ext)))
        image_files.extend(glob(os.path.join(input_dir, ext.upper())))
    
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print("画像ファイルが見つかりませんでした。")
        return

    print(f"対象枚数: {len(image_files)}枚")
    print(f"設定: KERNEL_SIZE={KERNEL_SIZE}, THRESHOLD={THRESHOLD}")
    print("-" * 30)

    for img_path in image_files:
        remove_marine_snow(img_path, output_dir)

    print("-" * 30)
    print("完了しました。")
    print("※ 'debug_masks' フォルダを確認し、白い魚などが誤って消えていないか確認してください。")
    print("※ 消しすぎならTHRESHOLDを上げ、残りすぎならKERNEL_SIZEを調整してください。")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"使用法: python {os.path.basename(sys.argv[0])} <入力画像フォルダ> <出力保存フォルダ>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(input_dir):
        print(f"エラー: 入力フォルダが見つかりません -> {input_dir}")
        sys.exit(1)

    process_folder(input_dir, output_dir)
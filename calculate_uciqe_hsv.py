import cv2
import numpy as np
import os
import glob
import argparse

def calculate_uciqe(img_bgr):
    """
    BGR形式の画像データからUCIQE値を計算する関数。

    Args:
        img_bgr (np.ndarray): BGR形式の画像データ。

    Returns:
        float: 計算されたUCIQE値。
    """
    # UCIQEの論文で提案されている重み係数
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576

    # BGRからCIELabへ色空間を変換
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # 輝度(L*)と色度(a*, b*)成分を分離
    l_channel = img_lab[:, :, 0].astype(np.float64)
    a_channel = img_lab[:, :, 1].astype(np.float64)
    b_channel = img_lab[:, :, 2].astype(np.float64)

    # 1. 色度の標準偏差 (σ_c)
    chroma = np.sqrt(np.square(a_channel) + np.square(b_channel))
    sigma_c = np.std(chroma)

    # 2. 輝度のコントラスト (con_l)
    # 上下1%のピクセルを除外して、外れ値の影響を低減
    l_flat = l_channel.flatten()
    l_flat_sorted = np.sort(l_flat)
    num_pixels = len(l_flat_sorted)
    percentile = int(num_pixels * 0.01)
    
    l_clipped = l_flat_sorted[percentile : num_pixels - percentile]
    con_l = np.max(l_clipped) - np.min(l_clipped)

    # 3. 彩度の平均値 (μ_s)
    # BGRからHSV色空間に変換し、Saturationチャネルを使用
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_channel = img_hsv[:, :, 1]
    mu_s = np.mean(s_channel)

    # UCIQEスコアを計算
    uciqe_score = c1 * sigma_c + c2 * con_l + c3 * mu_s
    
    return uciqe_score

def evaluate_images_in_folder(folder_path):
    """
    指定されたフォルダと全てのサブフォルダ内の全画像のUCIQE値を計算し、平均値を出力する関数。

    Args:
        folder_path (str): 評価対象の画像が含まれるルートフォルダのパス。
    """
    print(f"'{folder_path}' およびそのサブフォルダ内の画像を検索しています...")
    # 対応する画像拡張子
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for ext in image_extensions:
        # '**' を使うことで全てのサブディレクトリを対象にする
        # recursive=True が必要
        search_pattern = os.path.join(folder_path, '**', ext)
        image_paths.extend(glob.glob(search_pattern, recursive=True))

    if not image_paths:
        print(f"エラー: フォルダ '{folder_path}' 内に画像が見つかりません。")
        return

    scores = []
    print("\n--- 各画像のUCIQE値 ---")
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            score = calculate_uciqe(img)
            scores.append(score)
            # ルートフォルダからの相対パスを表示して分かりやすくする
            relative_path = os.path.relpath(path, folder_path)
            print(f"{relative_path}: {score:.4f}")
        else:
            print(f"警告: 画像ファイル '{path}' を読み込めませんでした。")
    
    if scores:
        average_score = np.mean(scores)
        print("\n--- 評価結果サマリー ---")
        print(f"処理した画像数: {len(scores)} 枚")
        print(f"UCIQE平均値: {average_score:.4f}")
    else:
        print("UCIQE値を計算できる画像がありませんでした。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="指定されたフォルダとサブフォルダ内の画像のUCIQE値を再帰的に計算し、その平均値を出力します。"
    )
    parser.add_argument(
        "folder_path", 
        type=str, 
        help="評価したい画像が含まれるルートフォルダのパス"
    )
    
    args = parser.parse_args()
    target_folder = args.folder_path

    print(f"評価対象フォルダ: {target_folder}")
    
    if os.path.isdir(target_folder):
        evaluate_images_in_folder(target_folder)
    else:
        print(f"エラー: 指定されたフォルダ '{target_folder}' が存在しません。")
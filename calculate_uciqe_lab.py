import cv2
import numpy as np
import os
import glob
import argparse

def calculate_uciqe_lab(img_bgr):
    """
    BGR形式の画像データからUCIQE値を計算する関数。
    0.0-1.0のレンジに正規化して計算し、一般的な論文の値(0.5前後)に合わせる。
    """
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576

    # 1. BGRからCIELabへ変換
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_lab = img_lab.astype(np.float64)

    # 2. 値を 0.0 - 1.0 の範囲に近い形に正規化する
    l_channel = img_lab[:, :, 0] / 255.0
    a_channel = (img_lab[:, :, 1] - 128.0) / 255.0
    b_channel = (img_lab[:, :, 2] - 128.0) / 255.0

    # --- 以下 計算ロジック ---

    # 1. 色度の標準偏差 (sigma_c)
    chroma = np.sqrt(np.square(a_channel) + np.square(b_channel))
    sigma_c = np.std(chroma)

    # 2. 輝度のコントラスト (con_l)
    l_flat = l_channel.flatten()
    l_flat_sorted = np.sort(l_flat)
    num_pixels = len(l_flat_sorted)
    percentile_count = int(num_pixels * 0.01)
    
    if percentile_count > 0:
        l_clipped = l_flat_sorted[percentile_count : -percentile_count]
    else:
        l_clipped = l_flat_sorted

    if len(l_clipped) > 0:
        con_l = np.max(l_clipped) - np.min(l_clipped)
    else:
        con_l = 0.0

    # 3. 彩度の平均値 (mu_s)
    with np.errstate(divide='ignore', invalid='ignore'):
        saturation = chroma / l_channel
        saturation[l_channel == 0] = 0
        saturation[~np.isfinite(saturation)] = 0
        
    mu_s = np.mean(saturation)

    # スコア計算
    uciqe_score = c1 * sigma_c + c2 * con_l + c3 * mu_s
    
    return uciqe_score

def evaluate_images_in_folder(folder_path):
    """
    指定されたフォルダと全てのサブフォルダ内の全画像のUCIQE値を計算し、
    平均値を出力するとともに、結果をテキストファイルに保存する関数。
    ※画像をタイムスタンプ順（古い順）に処理します。

    Args:
        folder_path (str): 評価対象の画像が含まれるルートフォルダのパス。
    """
    print(f"'{folder_path}' およびそのサブフォルダ内の画像を検索しています...")
    # 対応する画像拡張子
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_paths = []
    for ext in image_extensions:
        search_pattern = os.path.join(folder_path, '**', ext)
        image_paths.extend(glob.glob(search_pattern, recursive=True))

    if not image_paths:
        print(f"エラー: フォルダ '{folder_path}' 内に画像が見つかりません。")
        return

    # =========================================================
    # 変更箇所: 画像リストをタイムスタンプ（更新日時）順にソート
    # =========================================================
    try:
        image_paths.sort(key=os.path.getmtime)
        print("画像をタイムスタンプ順（古い順）に並べ替えました。")
    except Exception as e:
        print(f"ソート中に警告が発生しました（ファイル名順で処理します）: {e}")

    scores = []
    results_list = []
    
    print("\n--- 各画像のUCIQE値 (CIELab準拠 / 補正版) ---")
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            # 計算を実行
            score = calculate_uciqe_lab(img)
            scores.append(score)
            
            relative_path = os.path.relpath(path, folder_path)
            results_list.append((relative_path, score))
            
            # コンソールに出力
            print(f"{relative_path}: {score:.4f}")
        else:
            print(f"警告: 画像ファイル '{path}' を読み込めませんでした。")
    
    if scores:
        average_score = np.mean(scores)
        
        # --- コンソールへのサマリー出力 ---
        print("\n--- 評価結果サマリー ---")
        print(f"処理した画像数: {len(scores)} 枚")
        print(f"UCIQE平均値: {average_score:.4f}")

        # --- ファイルへの書き出し処理 ---
        output_filename = "uciqe_results.txt"
        output_path = os.path.join(folder_path, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("--- 各画像のUCIQE値 (CIELab準拠 / 補正版 / タイムスタンプ順) ---\n")
                for rel_path, score in results_list:
                    f.write(f"{rel_path}: {score:.4f}\n")
                
                f.write("\n--- 評価結果サマリー ---\n")
                f.write(f"処理した画像数: {len(scores)} 枚\n")
                f.write(f"UCIQE平均値: {average_score:.4f}\n")
            
            print(f"\n結果を {output_path} に保存しました。")
            
        except IOError as e:
            print(f"\nエラー: 結果ファイル '{output_path}' の書き込みに失敗しました。")
            print(f"詳細: {e}")

    else:
        print("UCIQE値を計算できる画像がありませんでした。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="指定されたフォルダとサブフォルダ内の画像のUCIQE値を再帰的に計算し、その平均値をコンソールに出力するとともに、"
                    "ルートフォルダに 'uciqe_results.txt' として保存します。(CIELab準拠 / 補正版 / タイムスタンプ順)"
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
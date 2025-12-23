"""
16bitのベイヤーTIF画像を16bitのカラーPNG画像に一括変換するスクリプト。

指定されたフォルダ内のTIF画像（*.tif, *.TIF）を読み込み、
デモザイク処理（カラー化）を行った後、16bit深度のPNG画像として保存する。

ファイル名に含まれる接尾辞 '_RC16' または '_LC16' に基づいて、
画像を 'RC' と 'LC' のサブフォルダに自動で振り分ける。

実行方法:
    python bayer_to_16bit_png.py <入力フォルダのパス>
"""

import glob
import os
import sys

import cv2 as cv
import numpy as np


def convert_and_sort(folder_path: str):
    """
    指定フォルダ内のベイヤーTIF画像をカラーPNGに変換し、振り分け保存する。

    1. 入力フォルダと同階層に、接尾辞 '_ColorPNG16' を持つ出力親フォルダと、
       その中に 'RC', 'LC' サブフォルダを作成する。
    2. 入力フォルダ内のTIF画像を16bit深度のまま読み込む。
    3. ベイヤーパターンをデモザイク処理し、16bitカラー画像（RGB）を生成する。
    4. ファイル名の接尾辞に基づき、'RC' または 'LC' フォルダに振り分ける。
    5. 16bit深度を維持したまま、PNG形式で画像を保存する。

    Args:
        folder_path (str): 処理対象のTIF画像が格納されたフォルダのパス。
    """
    if not os.path.isdir(folder_path):
        print(f"[ERROR] 指定されたフォルダが存在しません: {folder_path}")
        return

    # 出力フォルダ名を生成
    parent_dir = folder_path.rstrip("/\\") + "_ColorPNG16"
    rc_folder = os.path.join(parent_dir, "RC")
    lc_folder = os.path.join(parent_dir, "LC")

    # 出力フォルダを作成
    os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(rc_folder, exist_ok=True)
    os.makedirs(lc_folder, exist_ok=True)

    # 大文字・小文字両方の拡張子 (.tif, .TIF) に対応
    image_files = glob.glob(os.path.join(folder_path, "*.tif")) + \
        glob.glob(os.path.join(folder_path, "*.TIF"))
    image_files.sort()  # 処理順序を一定にするためファイル名でソート

    for file_path in image_files:
        # IMREAD_UNCHANGEDフラグで16bit (uint16) のまま読み込む
        bayer_img = cv.imread(file_path, cv.IMREAD_UNCHANGED)

        if bayer_img is None:
            print(f"[WARNING] ファイルの読み込みに失敗しました: {file_path}")
            continue

        print(
            f"処理中: {os.path.basename(file_path)} | "
            f"shape: {bayer_img.shape}, dtype: {bayer_img.dtype}"
        )

        if bayer_img.dtype != np.uint16:
            print(f"[WARNING] 16bit画像ではありません。スキップします: {file_path}")
            continue

        # ベイヤー変換（デモザイク処理）
        # 入力がuint16であれば、出力もuint16 (16bitカラー) となる
        try:
            rgb_img = cv.cvtColor(bayer_img, cv.COLOR_BayerBG2BGR)
        except cv.error as e:
            print(f"[ERROR] デモザイク処理中にエラーが発生しました {file_path}: {e}")
            print("  -> ベイヤー画像でないか、データ型が非対応の可能性があります。")
            continue

        # 保存ファイル名を設定 (拡張子を.pngに変更)
        base_name = os.path.basename(file_path)
        name, _ = os.path.splitext(base_name)
        filename = f"{name}.png"

        save_path = None  # 保存パスを初期化
        if name.endswith("_RC16"):
            save_path = os.path.join(rc_folder, filename)
        elif name.endswith("_LC16"):
            save_path = os.path.join(lc_folder, filename)
        else:
            print(f"[INFO] RC/LC接尾辞が見つからないためスキップします: {filename}")
            continue

        # PNG形式で保存
        # OpenCVはuint16型の画像を自動的に16bit PNGとして保存する
        try:
            cv.imwrite(save_path, rgb_img)
            print(f"-> 保存完了 (16bit PNG): {save_path}")
        except Exception as e:
            print(f"[ERROR] ファイルの保存に失敗しました {save_path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python bayer_to_16bit_png.py <folder_path>")
        sys.exit(1)

    input_folder = sys.argv[1]
    convert_and_sort(input_folder)
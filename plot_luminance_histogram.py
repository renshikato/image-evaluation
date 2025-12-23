import argparse
import sys
import cv2
import matplotlib.pyplot as plt


def main():
    """
    画像の輝度ヒストグラムをプロットする関数。
    ヒストグラムのピーク値を1.0として正規化し、
    軸の目盛りを指定した値（X軸:0,125,250 / Y軸:0.0,0.5,1.0）のみに設定します。
    """
    # --- フォントサイズの設定 ---
    TICK_SIZE = 32   # 目盛り数字（軸の数値）の大きさ
    # ------------------------

    # 引数の設定
    parser = argparse.ArgumentParser(description='画像の輝度ヒストグラムを表示します。')
    parser.add_argument('image_path', type=str, help='入力画像のパス')
    args = parser.parse_args()

    # 画像の読み込み
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"エラー: 画像が見つかりません: {args.image_path}")
        sys.exit(1)

    # 輝度画像への変換
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ヒストグラムの計算
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # 正規化（ピークを1.0にする）
    max_val = hist.max()
    if max_val > 0:
        hist_norm = hist / max_val
    else:
        hist_norm = hist

    # プロットの作成
    plt.figure(figsize=(10, 8))

    # 棒グラフとして描画
    plt.bar(range(256), hist_norm.ravel(), color='black', width=1.0)

    # 軸の範囲設定
    plt.xlim([0, 255])
    plt.ylim([0, 1.0])

    # --- 変更点: 目盛りの位置を特定の値に固定 ---
    # 横軸の目盛りを 0, 125, 250 に設定
    plt.xticks([0, 125, 250])
    
    # 縦軸の目盛りを 0.0, 0.5, 1.0 に設定
    plt.yticks([0.0, 0.5, 1.0])
    # ----------------------------------------

    # 目盛り数字のサイズ変更
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# --- グローバル変数 ---
drawing = False
ix, iy = -1, -1
img_display = None
img_raw = None
roi_data = []
g_image_path = ""

# --- フォントサイズ設定 ---
TITLE_SIZE = 24
SUB_TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 15
LEGEND_SIZE = 14

def save_coordinates_to_txt(data_list):
    """選択された領域の座標をテキストファイルに保存する"""
    if not g_image_path:
        return

    base_name = os.path.splitext(os.path.basename(g_image_path))[0]
    dir_name = os.path.dirname(g_image_path)
    output_filename = os.path.join(dir_name, f"{base_name}_coordinates.txt")

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"Target Image: {g_image_path}\n")
            f.write("=" * 40 + "\n")
            
            for i, (_, coords) in enumerate(data_list):
                x1, y1, x2, y2 = coords
                width = x2 - x1
                height = y2 - y1
                
                f.write(f"[Region {i+1}]\n")
                f.write(f"  Top-Left     : ({x1}, {y1})\n")
                f.write(f"  Bottom-Right : ({x2}, {y2})\n")
                f.write(f"  Size (WxH)   : {width} x {height}\n")
                f.write("-" * 40 + "\n")
                
        print(f">> 座標データを保存しました: {output_filename}")
        
    except Exception as e:
        print(f"Error: 座標ファイルの保存に失敗しました。{e}")

def calculate_radial_profile(magnitude_spectrum):
    """2次元スペクトルからラジアルプロファイル（円周平均）を計算"""
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    r_grid = np.sqrt((x - ccol)**2 + (y - crow)**2)
    r_int = r_grid.astype(int)
    tbin = np.bincount(r_int.ravel(), weights=magnitude_spectrum.ravel())
    nr = np.bincount(r_int.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    max_radius = min(crow, ccol)
    radial_profile = radial_profile[:max_radius]
    freq_axis = np.linspace(0, 0.5, num=max_radius)
    return freq_axis, radial_profile

def setup_axis_format(ax, xlabel, ylabel, title=None):
    """グラフの見た目を統一するためのヘルパー関数"""
    if title:
        ax.set_title(title, fontsize=SUB_TITLE_SIZE, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.grid(True, linestyle='--', alpha=0.5)

def analyze_two_rois(data_list):
    num_rois = len(data_list)
    if num_rois == 0: return

    # === Figure 1 & 2 の準備 ===
    fig1, axes1 = plt.subplots(2, 3, figsize=(20, 12))
    fig1.suptitle('Figure 1: Color & Brightness Analysis', fontsize=TITLE_SIZE)
    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
    fig2.suptitle('Figure 2: Frequency Domain Analysis', fontsize=TITLE_SIZE)

    for i in range(2):
        roi_img, _ = data_list[i]
        rows, cols = roi_img.shape[:2]
        
        # データ準備
        rgb_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        hsv_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        gray_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        region_label = "Region 1" if i == 0 else "Region 2"
        img_title = f"{region_label} ({cols}x{rows})"

        # --- Figure 1 描画 ---
        # 1-1. 元画像
        ax1_img = axes1[i, 0]
        ax1_img.imshow(rgb_img)
        ax1_img.set_title(img_title, fontsize=SUB_TITLE_SIZE, pad=10)
        ax1_img.axis('off')

        # 1-2. RGBヒストグラム
        ax1_hist = axes1[i, 1]
        colors = ('r', 'g', 'b')
        labels = ('Red', 'Green', 'Blue')
        for ch_idx, color in enumerate(colors):
            hist = cv2.calcHist([roi_img], [2-ch_idx], None, [256], [0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
            ax1_hist.plot(hist, color=color, linewidth=2.0, label=labels[ch_idx])
        ax1_hist.legend(loc='upper right', fontsize=LEGEND_SIZE, frameon=True)
        setup_axis_format(ax1_hist, 
                          xlabel="Pixel Value" if i == 1 else "", 
                          ylabel="Relative Frequency (Max = 1.0)" if i == 1 else "",
                          title="RGB Histogram" if i == 0 else None)
        ax1_hist.set_xlim([0, 256])
        ax1_hist.set_ylim([0, 1.05])

        # 1-3. HSV S-V Heatmap (背景白変更)
        ax1_sv = axes1[i, 2]
        s_vals = hsv_img[:, :, 1].ravel()
        v_vals = hsv_img[:, :, 2].ravel()
        H, xedges, yedges = np.histogram2d(s_vals, v_vals, bins=32, range=[[0, 256], [0, 256]])
        H_norm = H / H.max() if H.max() > 0 else H
        
        # === 変更点: 値が0の部分をマスク（透明化） ===
        H_masked = np.ma.masked_where(H_norm == 0, H_norm)
        
        # 背景色を白に明示的に設定
        ax1_sv.set_facecolor('white')
        
        im_sv = ax1_sv.imshow(H_masked.T, interpolation='nearest', origin='lower', 
                              extent=[0, 256, 0, 256], cmap='inferno', aspect='auto')
        
        setup_axis_format(ax1_sv, 
                          xlabel="Saturation" if i == 1 else "", 
                          ylabel="Value" if i == 1 else "",
                          title="HSV S-V Distribution" if i == 0 else None)
        cbar = fig1.colorbar(im_sv, ax=ax1_sv, fraction=0.046, pad=0.04)
        cbar.set_label('Relative Frequency (Max = 1.0)', fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

        # --- Figure 2 描画 ---
        # 2-1. 元画像(ROI)再掲
        ax2_img = axes2[i, 0]
        ax2_img.imshow(rgb_img)
        ax2_img.set_title(img_title, fontsize=SUB_TITLE_SIZE, pad=10)
        ax2_img.axis('off')

        # 2-2. 2D Spectrum
        data_f = gray_img.astype(float)
        fshift = np.fft.fftshift(np.fft.fft2(data_f))
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        freq_y = np.fft.fftshift(np.fft.fftfreq(rows, d=1.0))
        freq_x = np.fft.fftshift(np.fft.fftfreq(cols, d=1.0))
        extent = [freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()]
        
        ax2_spec = axes2[i, 1]
        ax2_spec.imshow(magnitude_spectrum, cmap='inferno', extent=extent, origin='lower')
        setup_axis_format(ax2_spec, 
                          xlabel="Frequency X" if i == 1 else "", 
                          ylabel="Frequency Y" if i == 1 else "",
                          title="2D Spectrum" if i == 0 else None)
        ticks = [-0.4, -0.2, 0.0, 0.2, 0.4]
        ax2_spec.set_xticks(ticks)
        ax2_spec.set_yticks(ticks)

        # 2-3. Radial Profile
        r_freq, r_profile = calculate_radial_profile(magnitude_spectrum)
        ax2_rad = axes2[i, 2]
        ax2_rad.plot(r_freq, r_profile, color='blue', linewidth=2.5)
        ax2_rad.set_xlim(0, 0.5)
        setup_axis_format(ax2_rad, 
                          xlabel="Frequency [cycle/pixel]" if i == 1 else "", 
                          ylabel="Magunitude (dB)" if i == 1 else "",
                          title="Radial Profile" if i == 0 else None)

    # レイアウト調整
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0, w_pad=3.0)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0, w_pad=3.0)

    # === Figure 3: 領域位置確認図 ===
    img_location = img_raw.copy()
    colors = [(0, 0, 255), (255, 0, 0)]

    for i in range(2):
        _, (x1, y1, x2, y2) = data_list[i]
        color = colors[i]
        label = f"Region {i+1}"
        cv2.rectangle(img_location, (x1, y1), (x2, y2), color, 3)
        text_y = max(y1 - 10, 30)
        cv2.putText(img_location, label, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.6, color, 5, cv2.LINE_AA)

    img_location_rgb = cv2.cvtColor(img_location, cv2.COLOR_BGR2RGB)
    
    fig3 = plt.figure(figsize=(12, 10))
    plt.imshow(img_location_rgb)
    plt.title('Figure 3: Selected Regions Location (R1:Red, R2:Blue)', fontsize=TITLE_SIZE)
    plt.axis('off')
    plt.tight_layout()

    print(">> 解析完了。3枚の図（Fig1, Fig2, Fig3）を表示します。")
    print("   ウィンドウが重なっている場合は移動させて確認してください。")
    plt.show()

# --- マウスイベント処理 ---
def on_mouse(event, x, y, flags, param):
    global ix, iy, drawing, img_display, roi_data, img_raw
    if len(roi_data) >= 2: return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = img_display.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Select 2 Regions', temp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_start, x_end = min(ix, x), max(ix, x)
        y_start, y_end = min(iy, y), max(iy, y)
        if x_end - x_start < 5 or y_end - y_start < 5:
            print("領域が小さすぎます。")
            return
        
        cv2.rectangle(img_display, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
        label = "R1" if len(roi_data) == 0 else "R2"
        cv2.putText(img_display, label, (x_start, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('Select 2 Regions', img_display)
        
        roi_img = img_raw[y_start:y_end, x_start:x_end]
        roi_coords = (x_start, y_start, x_end, y_end)
        roi_data.append((roi_img, roi_coords))
        
        print(f"[{len(roi_data)}/2] 選択完了")
        
        if len(roi_data) == 2:
            print(">> 解析開始...")
            save_coordinates_to_txt(roi_data)
            analyze_two_rois(roi_data)
            print("\n--- リセット ---")
            roi_data.clear()
            img_display = img_raw.copy()
            cv2.imshow('Select 2 Regions', img_display)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2領域比較解析 (Fig1:色彩, Fig2:周波数, Fig3:位置)')
    parser.add_argument('image_path', type=str, help='画像ファイルパス')
    args = parser.parse_args()
    
    g_image_path = args.image_path
    
    if not os.path.exists(g_image_path): sys.exit("File not found.")
    img_raw = cv2.imread(g_image_path)
    if img_raw is None: sys.exit("Load failed.")
    
    img_display = img_raw.copy()
    window_name = 'Select 2 Regions'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = img_raw.shape[:2]
    cv2.resizeWindow(window_name, w // 2, h // 2)
    cv2.setMouseCallback(window_name, on_mouse)
    
    print(f"Image loaded: {g_image_path}")
    print("【手順】")
    print("1. Region 1 (例: マリンスノー) を選択")
    print("2. Region 2 (例: 背景) を選択")
    while True:
        cv2.imshow(window_name, img_display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
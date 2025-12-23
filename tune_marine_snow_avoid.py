# ----------- ライブラリのインポート -----------
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import os
import sys
import numpy as np
from PIL import Image, ImageTk

# ----------- デフォルト設定値 -----------
DEFAULT_KERNEL_SIZE = 15   # 除去する粒の基準サイズ
DEFAULT_THRESHOLD = 20     # 感度（小さいほど薄いノイズも取る）
DEFAULT_INPAINT_R = 3      # 修復時の参照半径

# GUIアプリケーションのクラス
class MarineSnowTuner:
    def __init__(self, master):
        self.master = master
        self.master.title("マリンスノー除去 パラメータ調整ツール")
        self.master.geometry("1000x750")

        # --- 変数の初期化 ---
        self.image_path = None
        self.original_full_image = None  # 保存用の高解像度元画像
        self.preview_image = None        # 表示・調整用の縮小画像
        
        # パラメータ変数
        self.kernel_size = tk.IntVar(value=DEFAULT_KERNEL_SIZE)
        self.threshold = tk.IntVar(value=DEFAULT_THRESHOLD)
        self.inpaint_radius = tk.IntVar(value=DEFAULT_INPAINT_R)
        
        # 表示モード (0: 除去後画像, 1: 検出マスク)
        self.view_mode = tk.IntVar(value=0) 

        # --- GUIウィジェットの作成と配置 ---
        
        # 1. 上部フレーム (ファイル操作)
        top_frame = ttk.Frame(self.master)
        top_frame.pack(pady=10, fill=tk.X, padx=20)
        
        ttk.Button(top_frame, text="画像を開く", command=self.open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="結果を保存", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        # 表示モード切り替え（ラジオボタン）
        mode_frame = ttk.LabelFrame(top_frame, text="表示モード")
        mode_frame.pack(side=tk.RIGHT, padx=10)
        ttk.Radiobutton(mode_frame, text="除去結果 (Result)", variable=self.view_mode, value=0, command=self.update_image).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="検出マスク (Mask)", variable=self.view_mode, value=1, command=self.update_image).pack(side=tk.LEFT, padx=5)

        # 2. 中央フレーム (画像表示エリア)
        image_frame = ttk.Frame(self.master)
        image_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)

        # 左：元画像
        self.panel_left = ttk.LabelFrame(image_frame, text="元画像")
        self.panel_left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        self.label_original = ttk.Label(self.panel_left)
        self.label_original.pack(expand=True)

        # 右：処理後画像
        self.panel_right = ttk.LabelFrame(image_frame, text="プレビュー (処理結果)")
        self.panel_right.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        self.label_processed = ttk.Label(self.panel_right)
        self.label_processed.pack(expand=True)

        # 3. 下部フレーム (スライダーコントロール)
        control_frame = ttk.LabelFrame(self.master, text="パラメータ調整", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Gridの設定
        control_frame.columnconfigure(1, weight=1)

        # --- Kernel Size スライダー ---
        ttk.Label(control_frame, text="Kernel Size (粒の大きさ):").grid(row=0, column=0, sticky="w")
        # 奇数のみにしたいが、スライダーは連続値なのでupdate関数内で奇数に補正する
        self.scale_kernel = ttk.Scale(control_frame, from_=3, to=51, orient=tk.HORIZONTAL, 
                                      variable=self.kernel_size, command=self.update_image)
        self.scale_kernel.grid(row=0, column=1, sticky="ew", padx=10)
        self.lbl_kernel_val = ttk.Label(control_frame, text=f"{self.kernel_size.get()} px")
        self.lbl_kernel_val.grid(row=0, column=2, padx=5, sticky="w")

        # --- Threshold スライダー ---
        ttk.Label(control_frame, text="Threshold (感度/輝度差):").grid(row=1, column=0, sticky="w")
        self.scale_thresh = ttk.Scale(control_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                                      variable=self.threshold, command=self.update_image)
        self.scale_thresh.grid(row=1, column=1, sticky="ew", padx=10)
        self.lbl_thresh_val = ttk.Label(control_frame, text=f"{self.threshold.get()}")
        self.lbl_thresh_val.grid(row=1, column=2, padx=5, sticky="w")

        # --- Inpaint Radius スライダー ---
        ttk.Label(control_frame, text="Inpaint Radius (修復半径):").grid(row=2, column=0, sticky="w")
        self.scale_radius = ttk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, 
                                      variable=self.inpaint_radius, command=self.update_image)
        self.scale_radius.grid(row=2, column=1, sticky="ew", padx=10)
        self.lbl_radius_val = ttk.Label(control_frame, text=f"{self.inpaint_radius.get()} px")
        self.lbl_radius_val.grid(row=2, column=2, padx=5, sticky="w")

    def open_image(self):
        """画像を開く"""
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if not path:
            return
        
        self.image_path = path
        # 日本語パス対応のため numpy 経由で読む
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        self.original_full_image = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        stream.close()

        if self.original_full_image is None:
            messagebox.showerror("エラー", "画像を読み込めませんでした")
            return

        # ========================================================
        # 【修正箇所】16bit画像(uint16)の場合、8bit(uint8)に変換する
        # ========================================================
        if self.original_full_image.dtype == np.uint16:
            # 0-65535 を 0-255 にスケーリングして変換
            self.original_full_image = (self.original_full_image / 256).astype(np.uint8)
        # ========================================================

        # プレビュー用にリサイズ (処理速度向上のため)
        # 長辺を800px程度にする
        h, w = self.original_full_image.shape[:2]
        scale = 800 / max(h, w)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            self.preview_image = cv2.resize(self.original_full_image, (new_w, new_h))
        else:
            self.preview_image = self.original_full_image.copy()

        # 元画像を表示
        self.display_image(self.preview_image, self.label_original)
        
        # 処理実行
        self.update_image()

    def get_params(self):
        """現在のGUIパラメータを取得して整理する"""
        # カーネルサイズは奇数である必要がある
        k = int(self.kernel_size.get())
        if k % 2 == 0: k += 1
        
        t = int(self.threshold.get())
        r = int(self.inpaint_radius.get())
        return k, t, r

    def process_marine_snow(self, img, k, t, r, return_mask=False):
        """マリンスノー除去のコアロジック"""
        if img is None: return None

        # 1. グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. モルフォロジー・トップハット変換
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # 3. 2値化 (マスク作成)
        _, mask = cv2.threshold(tophat, t, 255, cv2.THRESH_BINARY)

        # 4. マスク膨張 (微調整)
        mask = cv2.dilate(mask, None, iterations=1)

        if return_mask:
            return mask

        # 5. インペイント
        result = cv2.inpaint(img, mask, r, cv2.INPAINT_TELEA)
        return result

    def update_image(self, event=None):
        """スライダー値をもとにプレビュー画像を更新"""
        if self.preview_image is None:
            return

        k, t, r = self.get_params()

        # ラベル更新
        self.lbl_kernel_val.config(text=f"{k} px")
        self.lbl_thresh_val.config(text=f"{t}")
        self.lbl_radius_val.config(text=f"{r} px")

        # モード確認
        is_mask_mode = (self.view_mode.get() == 1)

        # 処理実行
        if is_mask_mode:
            # マスク表示モード
            mask = self.process_marine_snow(self.preview_image, k, t, r, return_mask=True)
            # 表示用にRGB変換 (白黒)
            display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            self.panel_right.config(text="プレビュー (検出マスク - 白い部分が消去されます)")
        else:
            # 通常モード
            display_img = self.process_marine_snow(self.preview_image, k, t, r, return_mask=False)
            self.panel_right.config(text="プレビュー (処理結果)")

        self.display_image(display_img, self.label_processed)

    def display_image(self, cv_image, label_widget):
        """OpenCV画像をTkinterラベルに表示"""
        # 画像表示枠に合わせてリサイズするか、そのまま表示するか
        # ここでは既に preview_image がリサイズ済みなので、
        # さらにLabelに合わせて調整する処理を入れる
        img_h, img_w = cv_image.shape[:2]
        
        # BGR -> RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        tk_photo = ImageTk.PhotoImage(image=pil_image)
        
        label_widget.config(image=tk_photo)
        label_widget.image = tk_photo

    def save_image(self):
        """現在のパラメータで元画像(高画質)を処理して保存"""
        if self.original_full_image is None:
            return

        k, t, r = self.get_params()
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tif")],
            initialfile="cleaned_image.jpg"
        )
        
        if save_path:
            try:
                # 待機カーソル
                self.master.config(cursor="watch")
                self.master.update()

                # 高解像度画像に対して処理実行
                print(f"Saving... Kernel:{k}, Thresh:{t}, Radius:{r}")
                final_image = self.process_marine_snow(self.original_full_image, k, t, r)
                
                # 保存
                # 日本語パス対応
                is_success, im_buf = cv2.imencode(os.path.splitext(save_path)[1], final_image)
                if is_success:
                    with open(save_path, "wb") as f:
                        im_buf.tofile(f)
                    messagebox.showinfo("完了", f"保存しました:\n{save_path}")
                else:
                    messagebox.showerror("エラー", "保存に失敗しました")

            except Exception as e:
                messagebox.showerror("エラー", f"保存中にエラーが発生しました:\n{e}")
            finally:
                self.master.config(cursor="")

if __name__ == "__main__":
    root = tk.Tk()
    app = MarineSnowTuner(root)
    root.mainloop()
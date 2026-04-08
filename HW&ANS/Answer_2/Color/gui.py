"""
RPCA 图像修复 —— 交互式 GUI 模块

基于 Tkinter + Matplotlib，提供：
  - 图片选择
  - ALM 参数设置
  - RPCA 分解结果展示
"""

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from rpca_algorithm import ImageRestorer, detect_images

matplotlib.use("TkAgg")


# ============================================================
# GUI 主类
# ============================================================

class RPCAApp:
    """基于 Tkinter 的 RPCA 图像修复界面。"""

    BG_COLOR = "#f5f5f5"
    PANEL_COLOR = "#ffffff"
    SIDEBAR_WIDTH = 320

    def __init__(self, fig_root: str | Path):
        self.fig_root = Path(fig_root)
        self.images = detect_images(self.fig_root)

        self.restorer: ImageRestorer | None = None
        self.current_result = None

        self._build_ui()

        if self.images:
            self.image_var.set(self.images[0])
            self._on_image_selected()
        else:
            self._log("No images found in figs folder.")

    # ================================================================
    # UI 构建
    # ================================================================

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background=self.BG_COLOR)
        style.configure("Card.TFrame", background=self.PANEL_COLOR)
        style.configure("TLabel", background=self.BG_COLOR, font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(8, 5))
        style.configure("TCombobox", padding=4)
        style.configure("TLabelframe", background=self.BG_COLOR)
        style.configure("TLabelframe.Label", background=self.BG_COLOR, font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("RPCA Image Restoration")
        self.root.geometry("1400x850")
        self.root.configure(bg=self.BG_COLOR)

        self._status_var = tk.StringVar(master=self.root, value="Ready")

        self._configure_styles()

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0, minsize=self.SIDEBAR_WIDTH)
        self.root.rowconfigure(0, weight=1)

        canvas_frame = ttk.Frame(self.root, style="App.TFrame")
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(5, 0), pady=5)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        plot_frame = ttk.Frame(canvas_frame, style="Card.TFrame")
        plot_frame.grid(row=0, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        status_frame = ttk.Frame(canvas_frame, style="App.TFrame")
        status_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        self.fig = Figure(figsize=(10, 7), dpi=100, facecolor=self.BG_COLOR)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        ttk.Label(status_frame, textvariable=self._status_var, anchor="w").pack(fill=tk.X)

        sidebar = ttk.Frame(self.root, width=self.SIDEBAR_WIDTH, style="App.TFrame")
        sidebar.grid(row=0, column=1, sticky="nsew", padx=(0, 5), pady=5)
        sidebar.grid_propagate(False)

        self._build_sidebar(sidebar)

    def _build_sidebar(self, sidebar: ttk.Frame):
        px, py = 10, 6

        sec = ttk.LabelFrame(sidebar, text="Image", padding=8)
        sec.pack(fill=tk.X, padx=px, pady=py)

        self.image_var = tk.StringVar()
        self.image_cb = ttk.Combobox(
            sec,
            textvariable=self.image_var,
            values=self.images,
            state="readonly",
        )
        self.image_cb.pack(fill=tk.X, pady=(0, 8))
        self.image_cb.bind("<<ComboboxSelected>>", self._on_image_selected)

        btn_row = ttk.Frame(sec)
        btn_row.pack(fill=tk.X)

        ttk.Button(btn_row, text="Refresh", command=self._refresh_images).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4)
        )
        ttk.Button(btn_row, text="Load", command=self._on_image_selected).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0)
        )

        sec = ttk.LabelFrame(sidebar, text="ALM Parameters", padding=8)
        sec.pack(fill=tk.X, padx=px, pady=py)

        ttk.Label(sec, text="Lambda (blank = auto):").pack(anchor=tk.W)
        self.lambda_var = tk.StringVar(value="")
        ttk.Entry(sec, textvariable=self.lambda_var).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(sec, text="Tolerance:").pack(anchor=tk.W)
        self.tol_var = tk.StringVar(value="1e-7")
        ttk.Entry(sec, textvariable=self.tol_var).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(sec, text="Max Iterations:").pack(anchor=tk.W)
        self.max_iter_var = tk.StringVar(value="500")
        ttk.Entry(sec, textvariable=self.max_iter_var).pack(fill=tk.X, pady=(0, 8))

        btn_frame = ttk.Frame(sec)
        btn_frame.pack(fill=tk.X, pady=(4, 0))

        ttk.Button(btn_frame, text="Run RPCA", command=self._on_solve).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4)
        )
        ttk.Button(btn_frame, text="Reset", command=self._on_reset).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0)
        )

        out = ttk.LabelFrame(sidebar, text="Output", padding=8)
        out.pack(fill=tk.BOTH, expand=True, padx=px, pady=py)

        self.output_text = tk.Text(
            out,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#fafafa",
            relief=tk.FLAT,
            padx=8,
            pady=6,
        )
        scrollbar = ttk.Scrollbar(out, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    # ================================================================
    # 事件处理
    # ================================================================

    def _refresh_images(self):
        self.images = detect_images(self.fig_root)
        self.image_cb["values"] = self.images

        if self.images and self.image_var.get() not in self.images:
            self.image_var.set(self.images[0])

        self._log("Image list refreshed.")

    def _on_image_selected(self, _event=None):
        image_name = self.image_var.get()

        if not image_name:
            self._status_var.set("Please select an image")
            return

        image_path = self.fig_root / image_name
        if not image_path.exists():
            self._status_var.set("Image not found")
            self._log(f"Image not found: {image_path}")
            return

        self.restorer = ImageRestorer(image_path)
        self.current_result = None

        self._draw_original()
        self._status_var.set(f"Loaded {image_name}")
        self._log(f"Loaded image: {image_name}")

    def _on_solve(self):
        if self.restorer is None:
            self._status_var.set("Please load an image first")
            self._log("Please load an image first.")
            return

        try:
            lam = self._parse_lambda(self.lambda_var.get())
            tol = float(self.tol_var.get())
            max_iter = int(self.max_iter_var.get())
        except ValueError:
            self._status_var.set("Invalid parameter")
            self._log("Please enter valid parameter values.")
            return

        self._status_var.set("Running RPCA...")
        self.root.update_idletasks()

        result = self.restorer.solve(lam=lam, tol=tol, max_iter=max_iter)
        self.current_result = result

        self._draw_result(result)

        info = result["info"]
        self._status_var.set(
            "Done: iter = {}, error = {:.3e}".format(info["iterations"], info["error"])
        )

        self._log("-" * 40)
        self._log(f"Image      : {self.restorer.name}")
        self._log(f"Lambda     : {info['lambda']:.6f}")
        self._log(f"Iterations : {info['iterations']}")
        self._log(f"Final Error: {info['error']:.6e}")

        if isinstance(info["rank"], list):
            self._log(f"Rank(L)    : {info['rank']}")
        else:
            self._log(f"Rank(L)    : {info['rank']}")

        self._log(f"Sparsity(S): {info['sparse_ratio']:.4f}")
        self._log("-" * 40)

    def _on_reset(self):
        self.lambda_var.set("")
        self.tol_var.set("1e-7")
        self.max_iter_var.set("500")
        self.output_text.delete("1.0", tk.END)

        if self.restorer is not None:
            self._draw_original()
            self._status_var.set("Reset")
        else:
            self.fig.clear()
            self.canvas.draw_idle()
            self._status_var.set("Reset")

    # ================================================================
    # 绘图
    # ================================================================

    def _show_image(self, ax, image, title: str):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    def _draw_original(self):
        self.fig.clear()

        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        original = self.restorer.original

        self._show_image(ax1, original, "Original Image")
        self._show_image(ax2, original, "Input Image")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_result(self, result: dict):
        self.fig.clear()

        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)

        original = result["original"]
        input_image = result["input_image"]
        low_rank = result["low_rank"]
        sparse = result["sparse"]

        self._show_image(ax1, original, "Original Image")
        self._show_image(ax2, input_image, "Input Image")
        self._show_image(ax3, low_rank, "Low-rank Part L")
        self._show_image(ax4, sparse, "Sparse Part S")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ================================================================
    # 工具
    # ================================================================

    def _parse_lambda(self, text: str):
        text = text.strip()
        if text == "":
            return None
        return float(text)

    def _log(self, text: str):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def run(self):
        self.root.mainloop()
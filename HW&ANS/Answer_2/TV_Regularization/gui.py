"""
RPCA 图像修复 —— 高级 GUI 模块（TV Regularization）

基于 Tkinter + Matplotlib，提供：
  - 图片浏览与选择
  - RPCA / TV-RPCA 参数设置
  - 后台线程分解
  - 图像缩放、拖拽与结果导出
  - 分通道进度显示与结果信息展示
"""

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from rpca_algorithm import ImageRestorer, detect_images

matplotlib.use("TkAgg")


# ============================================================
# GUI 主类
# ============================================================

class RPCAApp:
    """基于 Tkinter 的 RPCA 图像修复界面。"""

    BG_COLOR = "#eef3f8"
    PANEL_COLOR = "#ffffff"
    CARD_COLOR = "#f8fafc"
    BORDER_COLOR = "#d7e2ec"
    ACCENT = "#284b63"
    ACCENT_SOFT = "#3c6e71"
    TITLE_COLOR = "#1b3448"
    TEXT_COLOR = "#243746"
    MUTED_TEXT = "#708090"
    SIDEBAR_WIDTH = 420
    ZOOM_MIN = 0.25
    ZOOM_MAX = 8.00
    WHEEL_STEP = 1.12
    DRAG_THRESHOLD = 3

    CHANNEL_STYLE = {
        "R": ("red.Horizontal.TProgressbar", "#9b2226"),
        "G": ("green.Horizontal.TProgressbar", "#2d6a4f"),
        "B": ("blue_channel.Horizontal.TProgressbar", "#1d4ed8"),
        "Gray": ("gray.Horizontal.TProgressbar", "#4b5563"),
    }

    def __init__(self, fig_root: str | Path):
        self.fig_root = Path(fig_root)
        self.current_dir = self.fig_root
        self.images = detect_images(self.current_dir)

        self.restorer: ImageRestorer | None = None
        self.current_result = None

        self.current_axes = []
        self._base_limits = []
        self._zoom_value = 1.0
        self._suspend_zoom_callback = False

        self._task_queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._busy = False

        self._loading_animation_job = None
        self._loading_base_text = "Idle"
        self._loading_phase = 0

        self._dragging = False
        self._drag_started = False
        self._drag_anchor_ax = None
        self._drag_start_pixels: tuple[float, float] | None = None
        self._drag_start_views: list[tuple[tuple[float, float], tuple[float, float]]] = []

        self._channel_order = ["R", "G", "B"]
        self._channel_rows = {}
        self._channel_label_vars = {}
        self._channel_state_vars = {}
        self._channel_detail_line1_vars = {}
        self._channel_detail_line2_vars = {}
        self._channel_bars = {}

        self._build_ui()
        self.root.after(120, self._poll_worker)

        if self.images:
            self.image_var.set(self.images[0])
            self._load_image_from_name(self.images[0])
        else:
            self._draw_empty("No image loaded")
            self._log("No images found in figs folder.")
            self._update_image_info()

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
        style.configure("Panel.TFrame", background=self.PANEL_COLOR)
        style.configure("Card.TFrame", background=self.CARD_COLOR, relief="solid", borderwidth=1)

        style.configure(
            "TLabel",
            background=self.BG_COLOR,
            foreground=self.TEXT_COLOR,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Panel.TLabel",
            background=self.PANEL_COLOR,
            foreground=self.TEXT_COLOR,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Info.TLabel",
            background=self.CARD_COLOR,
            foreground=self.TEXT_COLOR,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Muted.TLabel",
            background=self.CARD_COLOR,
            foreground=self.MUTED_TEXT,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Title.TLabel",
            background=self.BG_COLOR,
            foreground=self.TITLE_COLOR,
            font=("Segoe UI Semibold", 16),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BG_COLOR,
            foreground=self.MUTED_TEXT,
            font=("Segoe UI", 10),
        )

        style.configure("Section.TLabelframe", background=self.PANEL_COLOR, borderwidth=1, relief="solid")
        style.configure(
            "Section.TLabelframe.Label",
            background=self.PANEL_COLOR,
            foreground=self.ACCENT,
            font=("Segoe UI Semibold", 10),
        )

        style.configure("TButton", font=("Segoe UI", 10), padding=(8, 6), relief="flat")
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10), padding=(9, 6), foreground="#ffffff")
        style.map(
            "Accent.TButton",
            background=[("disabled", "#b7c9da"), ("active", self.ACCENT_SOFT), ("!disabled", self.ACCENT)],
            foreground=[("disabled", "#eef4f8"), ("!disabled", "#ffffff")],
        )
        style.map(
            "TButton",
            background=[("active", "#eef4fa"), ("!disabled", "#f5f8fb")],
            foreground=[("disabled", "#95a4b2"), ("!disabled", self.TEXT_COLOR)],
        )

        style.configure("TCombobox", padding=4)
        style.configure("Horizontal.TScale", background=self.BG_COLOR)
        style.configure("Vertical.TScrollbar", gripcount=0)

        style.configure(
            "red.Horizontal.TProgressbar",
            troughcolor="#f1f5f9",
            background=self.CHANNEL_STYLE["R"][1],
            bordercolor="#f1f5f9",
            lightcolor=self.CHANNEL_STYLE["R"][1],
            darkcolor=self.CHANNEL_STYLE["R"][1],
        )
        style.configure(
            "green.Horizontal.TProgressbar",
            troughcolor="#f1f5f9",
            background=self.CHANNEL_STYLE["G"][1],
            bordercolor="#f1f5f9",
            lightcolor=self.CHANNEL_STYLE["G"][1],
            darkcolor=self.CHANNEL_STYLE["G"][1],
        )
        style.configure(
            "blue_channel.Horizontal.TProgressbar",
            troughcolor="#f1f5f9",
            background=self.CHANNEL_STYLE["B"][1],
            bordercolor="#f1f5f9",
            lightcolor=self.CHANNEL_STYLE["B"][1],
            darkcolor=self.CHANNEL_STYLE["B"][1],
        )
        style.configure(
            "gray.Horizontal.TProgressbar",
            troughcolor="#f1f5f9",
            background=self.CHANNEL_STYLE["Gray"][1],
            bordercolor="#f1f5f9",
            lightcolor=self.CHANNEL_STYLE["Gray"][1],
            darkcolor=self.CHANNEL_STYLE["Gray"][1],
        )

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("RPCA Image Restoration")
        self.root.geometry("1560x940")
        self.root.configure(bg=self.BG_COLOR)
        self.root.minsize(1280, 760)

        self._status_var = tk.StringVar(master=self.root, value="Ready")
        self._loading_status_var = tk.StringVar(master=self.root, value="Idle")
        self._zoom_var = tk.DoubleVar(master=self.root, value=1.0)
        self._zoom_entry_var = tk.StringVar(master=self.root, value="1.00")

        self.path_var = tk.StringVar(master=self.root, value="-")
        self.size_var = tk.StringVar(master=self.root, value="-")
        self.channel_var = tk.StringVar(master=self.root, value="-")
        self.rank_var = tk.StringVar(master=self.root, value="-")
        self.sparse_var = tk.StringVar(master=self.root, value="-")
        self.tv_info_var = tk.StringVar(master=self.root, value="-")

        self._configure_styles()

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0, minsize=self.SIDEBAR_WIDTH)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=12)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        header = ttk.Frame(left, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(1, weight=1)

        title_block = ttk.Frame(header, style="App.TFrame")
        title_block.grid(row=0, column=0, sticky="w", padx=(0, 14))
        ttk.Label(title_block, text="Robust PCA Image Restoration", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title_block,
            text="Color RPCA with TV regularization and interactive visualization.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        self._build_zoom_toolbar(header)

        plot_panel = ttk.Frame(left, style="Panel.TFrame")
        plot_panel.grid(row=1, column=0, sticky="nsew")
        plot_panel.columnconfigure(0, weight=1)
        plot_panel.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(10, 7), dpi=100, facecolor=self.PANEL_COLOR)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("scroll_event", self._on_figure_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_figure_press)
        self.canvas.mpl_connect("button_release_event", self._on_figure_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_figure_motion)

        status_frame = ttk.Frame(left, style="App.TFrame")
        status_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        status_frame.columnconfigure(0, weight=1)
        ttk.Label(status_frame, textvariable=self._status_var, style="Panel.TLabel", anchor="w").grid(
            row=0,
            column=0,
            sticky="ew",
        )

        sidebar_shell = ttk.Frame(self.root, style="App.TFrame", width=self.SIDEBAR_WIDTH)
        sidebar_shell.grid(row=0, column=1, sticky="nsew", padx=(6, 12), pady=12)
        sidebar_shell.grid_propagate(False)
        sidebar_shell.columnconfigure(0, weight=1)
        sidebar_shell.rowconfigure(0, weight=1)

        self._build_scrollable_sidebar(sidebar_shell)

    def _build_zoom_toolbar(self, parent: ttk.Frame):
        bar = ttk.Frame(parent, style="App.TFrame")
        bar.grid(row=0, column=1, sticky="ew")
        bar.columnconfigure(3, weight=1)

        ttk.Button(bar, text="Zoom Out", command=lambda: self._set_zoom(self._zoom_value / 1.25)).grid(
            row=0, column=0, rowspan=2, padx=(0, 6), sticky="ns"
        )
        ttk.Button(bar, text="Fit View", command=self._reset_zoom).grid(
            row=0, column=1, rowspan=2, padx=(0, 10), sticky="ns"
        )

        scale_header = ttk.Frame(bar, style="App.TFrame")
        scale_header.grid(row=0, column=2, columnspan=2, sticky="ew")
        scale_header.columnconfigure(0, weight=1)

        ttk.Label(scale_header, text="Scale", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        self.zoom_entry = ttk.Entry(scale_header, textvariable=self._zoom_entry_var, width=8)
        self.zoom_entry.grid(row=0, column=1, sticky="e", padx=(10, 0))
        self.zoom_entry.bind("<Return>", self._on_zoom_entry)
        self.zoom_entry.bind("<FocusOut>", self._on_zoom_entry)

        scale = ttk.Scale(
            bar,
            orient=tk.HORIZONTAL,
            variable=self._zoom_var,
            from_=self.ZOOM_MIN,
            to=self.ZOOM_MAX,
            command=self._on_zoom_slider,
        )
        scale.grid(row=1, column=2, columnspan=2, sticky="ew", pady=(4, 0))

    def _build_scrollable_sidebar(self, parent: ttk.Frame):
        canvas = tk.Canvas(parent, bg=self.BG_COLOR, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.sidebar_canvas = canvas
        self.sidebar_frame = ttk.Frame(canvas, style="Panel.TFrame")
        self.sidebar_window = canvas.create_window((0, 0), window=self.sidebar_frame, anchor="nw")

        self.sidebar_frame.bind("<Configure>", self._on_sidebar_configure)
        self.sidebar_canvas.bind("<Configure>", self._on_sidebar_canvas_configure)
        self.sidebar_canvas.bind_all("<MouseWheel>", self._on_sidebar_mousewheel, add="+")
        self.sidebar_canvas.bind_all("<Button-4>", self._on_sidebar_mousewheel_linux, add="+")
        self.sidebar_canvas.bind_all("<Button-5>", self._on_sidebar_mousewheel_linux, add="+")

        self._build_sidebar_contents(self.sidebar_frame)

    def _build_sidebar_contents(self, sidebar: ttk.Frame):
        px, py = 10, 8

        source_sec = ttk.LabelFrame(sidebar, text="Image Source", style="Section.TLabelframe", padding=10)
        source_sec.pack(fill=tk.X, padx=px, pady=py)

        self.image_var = tk.StringVar()
        self.image_cb = ttk.Combobox(source_sec, textvariable=self.image_var, values=self.images, state="readonly")
        self.image_cb.pack(fill=tk.X, pady=(0, 8))
        self.image_cb.bind("<<ComboboxSelected>>", self._on_image_selected)

        source_row1 = ttk.Frame(source_sec, style="Panel.TFrame")
        source_row1.pack(fill=tk.X)
        source_row1.columnconfigure(0, weight=1)
        source_row1.columnconfigure(1, weight=1)

        self.browse_btn = ttk.Button(source_row1, text="Browse...", command=self._browse_image)
        self.browse_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.load_btn = ttk.Button(source_row1, text="Load", command=self._on_image_selected)
        self.load_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        source_row2 = ttk.Frame(source_sec, style="Panel.TFrame")
        source_row2.pack(fill=tk.X, pady=(8, 0))
        source_row2.columnconfigure(0, weight=1)
        source_row2.columnconfigure(1, weight=1)

        self.refresh_btn = ttk.Button(source_row2, text="Refresh Folder", command=self._refresh_images)
        self.refresh_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.save_btn = ttk.Button(source_row2, text="Save Figure", command=self._save_figure)
        self.save_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        param_sec = ttk.LabelFrame(sidebar, text="Model Parameters", style="Section.TLabelframe", padding=10)
        param_sec.pack(fill=tk.X, padx=px, pady=py)

        ttk.Label(param_sec, text="Lambda (blank = auto)", style="Panel.TLabel").pack(anchor=tk.W)
        self.lambda_var = tk.StringVar(value="")
        ttk.Entry(param_sec, textvariable=self.lambda_var).pack(fill=tk.X, pady=(2, 8))

        ttk.Label(param_sec, text="TV Weight", style="Panel.TLabel").pack(anchor=tk.W)
        self.tv_var = tk.StringVar(value="0.03")
        ttk.Entry(param_sec, textvariable=self.tv_var).pack(fill=tk.X, pady=(2, 8))

        ttk.Label(param_sec, text="Tolerance", style="Panel.TLabel").pack(anchor=tk.W)
        self.tol_var = tk.StringVar(value="1e-7")
        ttk.Entry(param_sec, textvariable=self.tol_var).pack(fill=tk.X, pady=(2, 8))

        ttk.Label(param_sec, text="Max Iterations", style="Panel.TLabel").pack(anchor=tk.W)
        self.max_iter_var = tk.StringVar(value="120")
        ttk.Entry(param_sec, textvariable=self.max_iter_var).pack(fill=tk.X, pady=(2, 10))

        run_row = ttk.Frame(param_sec, style="Panel.TFrame")
        run_row.pack(fill=tk.X)
        run_row.columnconfigure(0, weight=1)
        run_row.columnconfigure(1, weight=1)

        self.run_btn = ttk.Button(run_row, text="Run TV-RPCA", style="Accent.TButton", command=self._on_solve)
        self.run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.reset_btn = ttk.Button(run_row, text="Reset", command=self._on_reset)
        self.reset_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        comp_sec = ttk.LabelFrame(sidebar, text="Computation", style="Section.TLabelframe", padding=10)
        comp_sec.pack(fill=tk.X, padx=px, pady=py)

        loading_box = ttk.Frame(comp_sec, style="Card.TFrame")
        loading_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(loading_box, text="Status", style="Muted.TLabel").pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Label(loading_box, textvariable=self._loading_status_var, style="Info.TLabel").pack(
            anchor="w",
            padx=8,
            pady=(2, 6),
        )

        for name in self._channel_order:
            self._build_channel_progress_row(comp_sec, name)

        save_sec = ttk.LabelFrame(sidebar, text="Save Individual Panels", style="Section.TLabelframe", padding=10)
        save_sec.pack(fill=tk.X, padx=px, pady=py)

        save_grid = ttk.Frame(save_sec, style="Panel.TFrame")
        save_grid.pack(fill=tk.X)
        save_grid.columnconfigure(0, weight=1)
        save_grid.columnconfigure(1, weight=1)

        self.save_original_btn = ttk.Button(
            save_grid,
            text="Save Original",
            command=lambda: self._save_single_image("original", "original"),
            state=tk.DISABLED,
        )
        self.save_original_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 6))

        self.save_input_btn = ttk.Button(
            save_grid,
            text="Save Input",
            command=lambda: self._save_single_image("input_image", "input"),
            state=tk.DISABLED,
        )
        self.save_input_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 6))

        self.save_lowrank_btn = ttk.Button(
            save_grid,
            text="Save L",
            command=lambda: self._save_single_image("low_rank", "low_rank"),
            state=tk.DISABLED,
        )
        self.save_lowrank_btn.grid(row=1, column=0, sticky="ew", padx=(0, 4))

        self.save_sparse_btn = ttk.Button(
            save_grid,
            text="Save S",
            command=lambda: self._save_single_image("sparse", "sparse"),
            state=tk.DISABLED,
        )
        self.save_sparse_btn.grid(row=1, column=1, sticky="ew", padx=(4, 0))

        info_sec = ttk.LabelFrame(sidebar, text="Image Information", style="Section.TLabelframe", padding=10)
        info_sec.pack(fill=tk.X, padx=px, pady=py)

        self._build_info_block(info_sec, "Path", self.path_var)
        self._build_info_block(info_sec, "Size", self.size_var)
        self._build_info_block(info_sec, "Channels", self.channel_var)
        self._build_info_block(info_sec, "Rank(L)", self.rank_var)
        self._build_info_block(info_sec, "Sparsity(S)", self.sparse_var)
        self._build_info_block(info_sec, "TV Weight", self.tv_info_var)

        out_sec = ttk.LabelFrame(sidebar, text="Output", style="Section.TLabelframe", padding=10)
        out_sec.pack(fill=tk.BOTH, expand=True, padx=px, pady=py)

        self.output_text = tk.Text(
            out_sec,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg="#fcfdff",
            fg=self.TEXT_COLOR,
            relief=tk.FLAT,
            height=18,
            padx=8,
            pady=8,
            insertbackground=self.TEXT_COLOR,
        )
        output_scrollbar = ttk.Scrollbar(out_sec, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=output_scrollbar.set)
        output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def _build_channel_progress_row(self, parent: ttk.Frame, channel_name: str):
        row_box = ttk.Frame(parent, style="Card.TFrame")
        row_box.pack(fill=tk.X, pady=4)

        label_var = tk.StringVar(value=channel_name)
        state_var = tk.StringVar(value="Idle")
        detail_line1 = tk.StringVar(value="Iteration 0 / 0    Error --")
        detail_line2 = tk.StringVar(value="Rank --             Sparse --")

        header = ttk.Frame(row_box, style="Card.TFrame")
        header.pack(fill=tk.X, padx=8, pady=(6, 2))
        header.columnconfigure(0, weight=1)

        title_label = ttk.Label(
            header,
            textvariable=label_var,
            style="Info.TLabel",
            font=("Segoe UI Semibold", 10),
        )
        title_label.grid(row=0, column=0, sticky="w")

        state_label = ttk.Label(header, textvariable=state_var, style="Muted.TLabel")
        state_label.grid(row=0, column=1, sticky="e")

        style_name, color_value = self.CHANNEL_STYLE[channel_name]
        bar = ttk.Progressbar(row_box, mode="determinate", style=style_name)
        bar.pack(fill=tk.X, padx=8, pady=(0, 4))
        bar.configure(maximum=100, value=0)

        line1 = ttk.Label(row_box, textvariable=detail_line1, style="Muted.TLabel", justify=tk.LEFT)
        line1.pack(anchor="w", padx=8)
        line2 = ttk.Label(row_box, textvariable=detail_line2, style="Muted.TLabel", justify=tk.LEFT)
        line2.pack(anchor="w", padx=8, pady=(0, 6))

        self._channel_rows[channel_name] = row_box
        self._channel_label_vars[channel_name] = label_var
        self._channel_state_vars[channel_name] = state_var
        self._channel_detail_line1_vars[channel_name] = detail_line1
        self._channel_detail_line2_vars[channel_name] = detail_line2
        self._channel_bars[channel_name] = bar

    def _build_info_block(self, parent: ttk.Frame, label_text: str, value_var: tk.StringVar):
        box = ttk.Frame(parent, style="Card.TFrame")
        box.pack(fill=tk.X, pady=4)
        box.columnconfigure(0, weight=1)

        ttk.Label(box, text=label_text, style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=8, pady=(6, 0))
        ttk.Label(
            box,
            textvariable=value_var,
            style="Info.TLabel",
            justify=tk.LEFT,
            wraplength=self.SIDEBAR_WIDTH - 100,
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(2, 6))

    # ================================================================
    # 侧边栏滚动
    # ================================================================

    def _on_sidebar_configure(self, _event=None):
        self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

    def _on_sidebar_canvas_configure(self, event):
        self.sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width)

    def _on_sidebar_mousewheel(self, event):
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget is None:
            return
        if not self._is_descendant_of(widget, self.sidebar_frame):
            return
        delta = -1 if event.delta > 0 else 1
        self.sidebar_canvas.yview_scroll(delta, "units")

    def _on_sidebar_mousewheel_linux(self, event):
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        if widget is None:
            return
        if not self._is_descendant_of(widget, self.sidebar_frame):
            return
        delta = -1 if event.num == 4 else 1
        self.sidebar_canvas.yview_scroll(delta, "units")

    def _is_descendant_of(self, widget, ancestor) -> bool:
        cur = widget
        while cur is not None:
            if cur == ancestor:
                return True
            cur = cur.master
        return False

    # ================================================================
    # 图像加载与导出
    # ================================================================

    def _refresh_images(self):
        self.images = detect_images(self.current_dir)
        self.image_cb["values"] = self.images

        current_name = self.image_var.get()
        if self.images:
            if current_name in self.images:
                self.image_var.set(current_name)
            else:
                self.image_var.set(self.images[0])
        else:
            self.image_var.set("")

        self._log(f"Scanned folder: {self.current_dir}")

    def _browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            initialdir=str(self.current_dir),
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("PNG Files", "*.png"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("Bitmap Files", "*.bmp"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        path = Path(file_path)
        self.current_dir = path.parent
        self._refresh_images()
        self.image_var.set(path.name)
        self._load_image(path)

    def _save_figure(self):
        if not self.current_axes:
            messagebox.showinfo("Save Figure", "No figure is currently available.")
            return

        default_name = "rpca_result.png"
        if self.restorer is not None:
            default_name = f"{self.restorer.image_path.stem}_tv_rpca.png"

        file_path = filedialog.asksaveasfilename(
            title="Save current figure",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG", "*.png"),
                ("PDF", "*.pdf"),
                ("SVG", "*.svg"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        self.fig.savefig(file_path, dpi=220, bbox_inches="tight", facecolor=self.fig.get_facecolor())
        self._status_var.set(f"Figure saved to {file_path}")
        self._log(f"Saved figure: {file_path}")

    def _save_single_image(self, result_key: str, suffix: str):
        if self.current_result is None or result_key not in self.current_result:
            return

        default_stem = "rpca"
        if self.restorer is not None:
            default_stem = self.restorer.image_path.stem

        file_path = filedialog.asksaveasfilename(
            title="Save image panel",
            defaultextension=".png",
            initialfile=f"{default_stem}_{suffix}.png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("Bitmap", "*.bmp"),
                ("TIFF", "*.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        image = self.current_result[result_key]
        if image.ndim == 2:
            mpimg.imsave(file_path, image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            mpimg.imsave(file_path, image)

        self._status_var.set(f"Saved image: {file_path}")
        self._log(f"Saved image panel: {file_path}")

    def _set_result_save_buttons_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in [self.save_original_btn, self.save_input_btn, self.save_lowrank_btn, self.save_sparse_btn]:
            widget.configure(state=state)

    def _on_image_selected(self, _event=None):
        image_name = self.image_var.get()
        if not image_name:
            self._status_var.set("Please select an image")
            return
        self._load_image_from_name(image_name)

    def _load_image_from_name(self, image_name: str):
        image_path = self.current_dir / image_name
        if not image_path.exists():
            self._status_var.set("Image not found")
            self._log(f"Image not found: {image_path}")
            return
        self._load_image(image_path)

    def _load_image(self, image_path: Path):
        try:
            self.restorer = ImageRestorer(image_path)
        except Exception as exc:
            self._status_var.set("Failed to load image")
            self._log(f"Failed to load image: {exc}")
            messagebox.showerror("Load Image", f"Failed to load image:\n{exc}")
            return

        self.current_result = None
        self._set_result_save_buttons_state(False)
        self._draw_original()
        self._status_var.set(f"Loaded {image_path.name}")
        self._stop_loading_animation("Idle")
        self._reset_channel_progress_display()
        self._log(f"Loaded image: {image_path}")
        self._update_image_info()

    # ================================================================
    # 线程与进度显示
    # ================================================================

    def _on_solve(self):
        if self.restorer is None:
            self._status_var.set("Please load an image first")
            self._log("Please load an image first.")
            return
        if self._busy:
            self._log("A computation is already running.")
            return

        try:
            lam = self._parse_optional_float(self.lambda_var.get())
            tv_weight = float(self.tv_var.get())
            tol = float(self.tol_var.get())
            max_iter = int(self.max_iter_var.get())
        except ValueError:
            self._status_var.set("Invalid parameter")
            self._log("Please enter valid parameter values.")
            return

        if max_iter <= 0 or tv_weight < 0:
            self._status_var.set("Invalid parameter")
            self._log("Max Iterations must be positive and TV Weight must be nonnegative.")
            return

        self._set_busy(True)
        self._status_var.set("Running TV-RPCA in background...")
        self._start_loading_animation("Computing")
        self._reset_channel_progress_display(max_iter=max_iter)
        self._log("Starting TV-RPCA computation...")

        image_path = self.restorer.image_path

        def progress_callback(channel_index, channel_name, iter_idx, total_iter, error, rank, sparse_ratio):
            self._task_queue.put(
                (
                    "progress",
                    {
                        "channel_index": channel_index,
                        "channel_name": channel_name,
                        "iter": iter_idx,
                        "max_iter": total_iter,
                        "error": error,
                        "rank": rank,
                        "sparse_ratio": sparse_ratio,
                    },
                )
            )

        def worker():
            try:
                restorer = ImageRestorer(image_path)
                result = restorer.solve(
                    lam=lam,
                    tv_weight=tv_weight,
                    tol=tol,
                    max_iter=max_iter,
                    tv_inner_iter=20,
                    progress_callback=progress_callback,
                )
                self._task_queue.put(("success", result))
            except Exception as exc:
                self._task_queue.put(("error", str(exc)))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _poll_worker(self):
        try:
            while True:
                kind, payload = self._task_queue.get_nowait()
                if kind == "progress":
                    self._update_channel_progress(
                        channel_name=payload["channel_name"],
                        iteration=payload["iter"],
                        max_iter=payload["max_iter"],
                        error=payload["error"],
                        rank=payload["rank"],
                        sparse_ratio=payload["sparse_ratio"],
                        state_text="Running",
                    )
                elif kind == "success":
                    self.current_result = payload
                    self._draw_result(payload)
                    self._set_result_save_buttons_state(True)
                    self._update_image_info()

                    info = payload["info"]
                    for item in info.get("channel_infos", []):
                        self._update_channel_progress(
                            channel_name=item["channel_name"],
                            iteration=item["iterations"],
                            max_iter=max(int(self.max_iter_var.get()), item["iterations"]),
                            error=item["error"],
                            rank=item["rank"],
                            sparse_ratio=item["sparse_ratio"],
                            state_text="Finished",
                        )

                    self._status_var.set(
                        "Done: iter = {}, error = {:.3e}".format(info["iterations"], info["error"])
                    )
                    self._stop_loading_animation("Finished")

                    self._log("-" * 40)
                    self._log(f"Image      : {self.restorer.name}")
                    self._log(f"Lambda     : {info['lambda']:.6f}")
                    self._log(f"TV Weight  : {info['tv_weight']:.6f}")
                    self._log(f"Iterations : {info['iterations']}")
                    self._log(f"Final Error: {info['error']:.6e}")
                    self._log(f"Rank(L)    : {info['rank']}")
                    self._log(f"Sparsity(S): {info['sparse_ratio']:.4f}")
                    self._log("-" * 40)

                    self._set_busy(False)
                else:
                    self._status_var.set("TV-RPCA failed")
                    self._stop_loading_animation("Failed")
                    self._log(f"TV-RPCA failed: {payload}")
                    messagebox.showerror("TV-RPCA", f"TV-RPCA failed:\n{payload}")
                    self._set_busy(False)
        except queue.Empty:
            pass

        self.root.after(120, self._poll_worker)

    def _set_busy(self, busy: bool):
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL

        for widget in [self.run_btn, self.load_btn, self.browse_btn, self.refresh_btn, self.save_btn, self.reset_btn]:
            widget.configure(state=state)

        self.image_cb.configure(state="disabled" if busy else "readonly")
        self.run_btn.configure(text="Running..." if busy else "Run TV-RPCA")

    def _start_loading_animation(self, base_text: str):
        self._loading_base_text = base_text
        self._loading_phase = 0
        self._animate_loading_text()

    def _animate_loading_text(self):
        if not self._busy:
            return
        dots = "." * (self._loading_phase % 4)
        self._loading_status_var.set(f"{self._loading_base_text}{dots}")
        self._loading_phase += 1
        self._loading_animation_job = self.root.after(300, self._animate_loading_text)

    def _stop_loading_animation(self, final_text: str):
        if self._loading_animation_job is not None:
            self.root.after_cancel(self._loading_animation_job)
            self._loading_animation_job = None
        self._loading_status_var.set(final_text)

    def _reset_channel_progress_display(self, max_iter: int = 0):
        channel_mode = self._infer_channel_mode()

        if channel_mode == "Gray":
            self._channel_rows["R"].pack_forget()
            self._channel_rows["G"].pack_forget()
            self._channel_rows["B"].pack_forget()
            gray_row = self._channel_rows["R"]
            gray_row.pack(fill=tk.X, pady=4)
            self._channel_label_vars["R"].set("Gray")
            self._channel_state_vars["R"].set("Idle")
            self._channel_detail_line1_vars["R"].set(f"Iteration 0 / {max_iter}")
            self._channel_detail_line2_vars["R"].set("Rank --             Sparse --")
            bar = self._channel_bars["R"]
            bar.configure(style=self.CHANNEL_STYLE["Gray"][0], maximum=max(max_iter, 1), value=0)
            return

        for name in self._channel_order:
            row = self._channel_rows[name]
            if not row.winfo_manager():
                row.pack(fill=tk.X, pady=4)
            self._channel_label_vars[name].set(name)
            self._channel_state_vars[name].set("Idle")
            self._channel_detail_line1_vars[name].set(f"Iteration 0 / {max_iter}")
            self._channel_detail_line2_vars[name].set("Rank --             Sparse --")
            bar = self._channel_bars[name]
            bar.configure(style=self.CHANNEL_STYLE[name][0], maximum=max(max_iter, 1), value=0)

    def _update_channel_progress(
        self,
        channel_name: str,
        iteration: int,
        max_iter: int,
        error: float,
        rank,
        sparse_ratio: float,
        state_text: str,
    ):
        key = "R" if channel_name == "Gray" else channel_name
        if key not in self._channel_bars:
            return

        self._channel_label_vars[key].set(channel_name)
        self._channel_state_vars[key].set(state_text)
        self._channel_detail_line1_vars[key].set(
            f"Iteration {iteration} / {max_iter}    Error {error:.3e}"
        )
        self._channel_detail_line2_vars[key].set(
            f"Rank {rank}             Sparse {sparse_ratio:.4f}"
        )
        bar = self._channel_bars[key]
        bar.configure(maximum=max(max_iter, 1), value=min(max(iteration, 0), max(max_iter, 1)))

    def _infer_channel_mode(self) -> str:
        if self.restorer is None:
            return "RGB"
        return "Gray" if self.restorer.original.ndim == 2 else "RGB"

    # ================================================================
    # 缩放与拖拽
    # ================================================================

    def _reset_zoom(self):
        self._set_zoom(1.0)

    def _on_zoom_slider(self, value):
        if self._suspend_zoom_callback:
            return
        self._set_zoom(float(value), update_controls=False)

    def _on_zoom_entry(self, _event=None):
        try:
            value = float(self._zoom_entry_var.get())
        except ValueError:
            self._zoom_entry_var.set(f"{self._zoom_value:.2f}")
            return
        self._set_zoom(value)

    def _set_zoom(self, zoom_value: float, update_controls: bool = True):
        if not self.current_axes or not self._base_limits:
            self._sync_zoom_controls(1.0)
            return

        zoom_value = max(self.ZOOM_MIN, min(self.ZOOM_MAX, float(zoom_value)))
        self._zoom_value = zoom_value

        for ax, limits in zip(self.current_axes, self._base_limits):
            self._apply_axis_zoom(ax, limits, zoom_value)

        self.canvas.draw_idle()
        if update_controls:
            self._sync_zoom_controls(zoom_value)
        else:
            self._zoom_entry_var.set(f"{zoom_value:.2f}")

    def _sync_zoom_controls(self, zoom_value: float):
        self._suspend_zoom_callback = True
        self._zoom_var.set(zoom_value)
        self._suspend_zoom_callback = False
        self._zoom_entry_var.set(f"{zoom_value:.2f}")

    def _apply_axis_zoom(self, ax, limits, zoom_value: float):
        x0, x1, y0, y1 = limits
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        width = (x1 - x0) / zoom_value
        height = (y0 - y1) / zoom_value
        ax.set_xlim(cx - width / 2, cx + width / 2)
        ax.set_ylim(cy + height / 2, cy - height / 2)

    def _on_figure_scroll(self, event):
        if not self.current_axes or event.inaxes is None:
            return

        factor = self.WHEEL_STEP if event.button == "up" else 1.0 / self.WHEEL_STEP
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom_value * factor))
        ratio = self._zoom_value / new_zoom

        for ax, limits in zip(self.current_axes, self._base_limits):
            if ax == event.inaxes and event.xdata is not None and event.ydata is not None:
                self._apply_axis_zoom_at_anchor(ax, limits, new_zoom, event.xdata, event.ydata, ratio)
            else:
                self._apply_axis_zoom(ax, limits, new_zoom)

        self._zoom_value = new_zoom
        self._sync_zoom_controls(new_zoom)
        self.canvas.draw_idle()

    def _apply_axis_zoom_at_anchor(self, ax, limits, zoom_value: float, anchor_x: float, anchor_y: float, ratio: float):
        cur_x0, cur_x1 = ax.get_xlim()
        cur_y0, cur_y1 = ax.get_ylim()

        width = (cur_x1 - cur_x0) * ratio
        height = (cur_y0 - cur_y1) * ratio

        if abs(cur_x1 - cur_x0) < 1e-12 or abs(cur_y0 - cur_y1) < 1e-12:
            self._apply_axis_zoom(ax, limits, zoom_value)
            return

        relx = (anchor_x - cur_x0) / (cur_x1 - cur_x0)
        rely = (anchor_y - cur_y1) / (cur_y0 - cur_y1)

        new_x0 = anchor_x - relx * width
        new_x1 = new_x0 + width
        new_y1 = anchor_y - rely * height
        new_y0 = new_y1 + height

        ax.set_xlim(new_x0, new_x1)
        ax.set_ylim(new_y0, new_y1)
        self._clip_axis_to_base(ax, limits)

    def _on_figure_press(self, event):
        if event.button != 1 or event.inaxes not in self.current_axes:
            return
        if event.x is None or event.y is None:
            return

        self._dragging = True
        self._drag_started = False
        self._drag_anchor_ax = event.inaxes
        self._drag_start_pixels = (event.x, event.y)
        self._drag_start_views = []
        for ax in self.current_axes:
            self._drag_start_views.append((ax.get_xlim(), ax.get_ylim()))

    def _on_figure_release(self, _event):
        self._dragging = False
        self._drag_started = False
        self._drag_anchor_ax = None
        self._drag_start_pixels = None
        self._drag_start_views = []

    def _on_figure_motion(self, event):
        if not self._dragging or self._drag_anchor_ax is None:
            return
        if event.x is None or event.y is None or self._drag_start_pixels is None:
            return

        start_x, start_y = self._drag_start_pixels
        dx_pix = event.x - start_x
        dy_pix = event.y - start_y

        if not self._drag_started:
            if abs(dx_pix) < self.DRAG_THRESHOLD and abs(dy_pix) < self.DRAG_THRESHOLD:
                return
            self._drag_started = True

        bbox = self._drag_anchor_ax.bbox
        if bbox.width <= 1 or bbox.height <= 1:
            return

        anchor_idx = self.current_axes.index(self._drag_anchor_ax)
        start_xlim, start_ylim = self._drag_start_views[anchor_idx]
        dx_data = dx_pix * (start_xlim[1] - start_xlim[0]) / bbox.width
        dy_data = dy_pix * (start_ylim[1] - start_ylim[0]) / bbox.height

        for i, ax in enumerate(self.current_axes):
            xlim0, ylim0 = self._drag_start_views[i]
            ax.set_xlim(xlim0[0] - dx_data, xlim0[1] - dx_data)
            ax.set_ylim(ylim0[0] - dy_data, ylim0[1] - dy_data)
            self._clip_axis_to_base(ax, self._base_limits[i])

        self.canvas.draw_idle()

    def _clip_axis_to_base(self, ax, limits):
        base_x0, base_x1, base_y0, base_y1 = limits
        cur_x0, cur_x1 = ax.get_xlim()
        cur_y0, cur_y1 = ax.get_ylim()

        width = cur_x1 - cur_x0
        height = cur_y0 - cur_y1
        base_width = base_x1 - base_x0
        base_height = base_y0 - base_y1

        if width >= base_width or height >= base_height:
            ax.set_xlim(base_x0, base_x1)
            ax.set_ylim(base_y0, base_y1)
            return

        if cur_x0 < base_x0:
            cur_x0 = base_x0
            cur_x1 = cur_x0 + width
        if cur_x1 > base_x1:
            cur_x1 = base_x1
            cur_x0 = cur_x1 - width

        if cur_y0 > base_y0:
            cur_y0 = base_y0
            cur_y1 = cur_y0 - height
        if cur_y1 < base_y1:
            cur_y1 = base_y1
            cur_y0 = cur_y1 + height

        ax.set_xlim(cur_x0, cur_x1)
        ax.set_ylim(cur_y0, cur_y1)

    # ================================================================
    # 绘图
    # ================================================================

    def _draw_empty(self, title: str):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#ffffff")
        ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=14, color=self.MUTED_TEXT)
        ax.axis("off")
        self.current_axes = [ax]
        self._capture_base_limits()
        self.canvas.draw_idle()
        self._sync_zoom_controls(1.0)

    def _show_image(self, ax, image, title: str):
        if image.ndim == 2:
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=11, color=self.ACCENT, pad=8)
        ax.axis("off")

    def _draw_original(self):
        self.fig.clear()

        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        original = self.restorer.original

        self._show_image(ax1, original, "Original Image")
        self._show_image(ax2, original, "Input Image")

        self.current_axes = [ax1, ax2]
        self.fig.tight_layout(pad=2.1)
        self._capture_base_limits()
        self.canvas.draw_idle()
        self._reset_zoom()

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

        self.current_axes = [ax1, ax2, ax3, ax4]
        self.fig.tight_layout(pad=2.1)
        self._capture_base_limits()
        self.canvas.draw_idle()
        self._reset_zoom()

    def _capture_base_limits(self):
        self._base_limits = []
        for ax in self.current_axes:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            self._base_limits.append((x0, x1, max(y0, y1), min(y0, y1)))

    # ================================================================
    # 信息与日志
    # ================================================================

    def _update_image_info(self):
        if self.restorer is None:
            self.path_var.set("-")
            self.size_var.set("-")
            self.channel_var.set("-")
            self.rank_var.set("-")
            self.sparse_var.set("-")
            self.tv_info_var.set("-")
            return

        image = self.restorer.original
        h, w = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]

        self.path_var.set(str(self.restorer.image_path))
        self.size_var.set(f"{w} × {h}")
        self.channel_var.set(str(channels))

        if self.current_result is None:
            self.rank_var.set("-")
            self.sparse_var.set("-")
            self.tv_info_var.set(self.tv_var.get().strip() if self.tv_var.get().strip() else "0")
            return

        info = self.current_result["info"]
        self.rank_var.set(str(info.get("rank", "-")))
        self.sparse_var.set(f"{info.get('sparse_ratio', 0.0):.4f}")
        self.tv_info_var.set(f"{info.get('tv_weight', 0.0):.6f}")

    def _log(self, text: str):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    # ================================================================
    # 工具
    # ================================================================

    def _parse_optional_float(self, text: str):
        text = text.strip()
        if text == "":
            return None
        return float(text)

    def _on_reset(self):
        if self._busy:
            return

        self.lambda_var.set("")
        self.tv_var.set("0.03")
        self.tol_var.set("1e-7")
        self.max_iter_var.set("120")
        self.output_text.delete("1.0", tk.END)

        if self.restorer is not None:
            self.current_result = None
            self._set_result_save_buttons_state(False)
            self._draw_original()
        else:
            self._draw_empty("No image loaded")

        self._status_var.set("Reset")
        self._stop_loading_animation("Idle")
        self._reset_channel_progress_display()
        self._update_image_info()

    def run(self):
        self.root.mainloop()
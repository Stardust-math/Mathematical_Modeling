"""
RPCA 图像修复算法模块 —— 图像读取、ALM 分解、结果封装
"""

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


# ============================================================
# 数据读取
# ============================================================

def detect_images(fig_root: str | Path) -> list[str]:
    """扫描 figs 文件夹中的图片文件。"""
    fig_root = Path(fig_root)
    image_names = []

    for pattern in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
        for path in sorted(fig_root.glob(pattern)):
            image_names.append(path.name)

    return sorted(image_names)


def read_image(image_path: str | Path) -> np.ndarray:
    """读取图片，并返回归一化到 [0, 1] 的图像矩阵。"""
    image_path = Path(image_path)
    img = mpimg.imread(image_path)

    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float64) / 255.0
    else:
        img = img.astype(np.float64)
        img = np.clip(img, 0.0, 1.0)

    return img


# ============================================================
# RPCA 核心算子
# ============================================================

def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """逐元素软阈值。"""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def singular_value_threshold(X: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
    """奇异值软阈值。"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_shrink = np.maximum(s - tau, 0.0)
    rank = int(np.sum(s_shrink > 0))

    if rank == 0:
        return np.zeros_like(X), 0

    L = (U[:, :rank] * s_shrink[:rank]) @ Vt[:rank, :]
    return L, rank


# ============================================================
# ALM 求解 RPCA
# ============================================================

def rpca_alm(
    A: np.ndarray,
    lam: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 500,
    progress_callback=None,
    channel_index: int | None = None,
    channel_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    使用 Inexact ALM 求解：

        min ||L||_* + lambda ||S||_1
        s.t. A = L + S
    """
    A = A.astype(np.float64)
    m, n = A.shape

    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    norm_two = np.linalg.norm(A, 2)
    norm_inf = np.max(np.abs(A)) / max(lam, 1e-12)
    dual_norm = max(norm_two, norm_inf, 1e-12)

    Y = A / dual_norm
    L = np.zeros_like(A)
    S = np.zeros_like(A)

    mu = 1.25 / max(norm_two, 1e-12)
    mu_bar = mu * 1e7
    rho = 1.5

    norm_A = np.linalg.norm(A, ord="fro")
    if norm_A < 1e-12:
        norm_A = 1.0

    history = []
    rank = 0

    for it in range(1, max_iter + 1):
        L, rank = singular_value_threshold(A - S + Y / mu, 1.0 / mu)
        S = soft_threshold(A - L + Y / mu, lam / mu)

        residual = A - L - S
        err = np.linalg.norm(residual, ord="fro") / norm_A

        Y = Y + mu * residual
        mu = min(mu * rho, mu_bar)

        sparse_ratio = float(np.mean(np.abs(S) > 1e-3))
        history.append(
            {
                "iter": it,
                "error": err,
                "rank": rank,
                "sparse_ratio": sparse_ratio,
            }
        )

        if progress_callback is not None:
            progress_callback(
                channel_index=channel_index,
                channel_name=channel_name,
                iter_idx=it,
                total_iter=max_iter,
                error=err,
                rank=rank,
                sparse_ratio=sparse_ratio,
            )

        if err < tol:
            break

    info = {
        "iterations": history[-1]["iter"],
        "error": history[-1]["error"],
        "rank": history[-1]["rank"],
        "sparse_ratio": history[-1]["sparse_ratio"],
        "lambda": lam,
        "history": history,
        "channel_index": channel_index,
        "channel_name": channel_name,
    }

    return L, S, info


def rpca_alm_color(
    A: np.ndarray,
    lam: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 500,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    对彩图的每个通道分别执行 RPCA。
    若输入本身是灰度图，则退化为单通道 RPCA。
    """
    if A.ndim == 2:
        return rpca_alm(
            A,
            lam=lam,
            tol=tol,
            max_iter=max_iter,
            progress_callback=progress_callback,
            channel_index=0,
            channel_name="Gray",
        )

    if A.ndim != 3 or A.shape[2] != 3:
        raise ValueError("Input image must be grayscale or RGB.")

    L_channels = []
    S_channels = []
    channel_infos = []
    channel_names = ["R", "G", "B"]

    for c, channel_name in enumerate(channel_names):
        Lc, Sc, info_c = rpca_alm(
            A[:, :, c],
            lam=lam,
            tol=tol,
            max_iter=max_iter,
            progress_callback=progress_callback,
            channel_index=c,
            channel_name=channel_name,
        )
        L_channels.append(Lc)
        S_channels.append(Sc)
        channel_infos.append(info_c)

    L = np.stack(L_channels, axis=2)
    S = np.stack(S_channels, axis=2)

    info = {
        "iterations": max(item["iterations"] for item in channel_infos),
        "error": max(item["error"] for item in channel_infos),
        "rank": [item["rank"] for item in channel_infos],
        "sparse_ratio": float(np.mean([item["sparse_ratio"] for item in channel_infos])),
        "lambda": channel_infos[0]["lambda"],
        "channel_infos": channel_infos,
    }

    return L, S, info


# ============================================================
# 结果显示辅助
# ============================================================

def normalize_to_unit(X: np.ndarray) -> np.ndarray:
    """将矩阵线性映射到 [0, 1]，用于显示。"""
    X = X.astype(np.float64)

    if X.ndim == 2:
        x_min = float(np.min(X))
        x_max = float(np.max(X))
        if abs(x_max - x_min) < 1e-12:
            return np.zeros_like(X)
        return (X - x_min) / (x_max - x_min)

    if X.ndim == 3:
        Y = np.zeros_like(X)
        for c in range(X.shape[2]):
            channel = X[:, :, c]
            c_min = float(np.min(channel))
            c_max = float(np.max(channel))
            if abs(c_max - c_min) < 1e-12:
                Y[:, :, c] = 0.0
            else:
                Y[:, :, c] = (channel - c_min) / (c_max - c_min)
        return Y

    raise ValueError("Input must be a 2D or 3D array.")


def sparse_to_display(S: np.ndarray) -> np.ndarray:
    """将稀疏项转成便于显示的图像。"""
    return normalize_to_unit(np.abs(S))


# ============================================================
# 高层封装
# ============================================================

class ImageRestorer:
    """封装单张图片的读取与 RPCA 修复。"""

    def __init__(self, image_path: str | Path):
        image_path = Path(image_path)
        self.image_path = image_path
        self.name = image_path.name

        self.original = read_image(image_path)

    def solve(
        self,
        lam: float | None = None,
        tol: float = 1e-7,
        max_iter: int = 500,
        progress_callback=None,
    ) -> dict:
        """执行 RPCA 分解。"""
        L, S, info = rpca_alm_color(
            self.original,
            lam=lam,
            tol=tol,
            max_iter=max_iter,
            progress_callback=progress_callback,
        )

        result = {
            "original": self.original,
            "input_image": self.original,
            "low_rank": np.clip(L, 0.0, 1.0),
            "sparse": sparse_to_display(S),
            "reconstruction": np.clip(L + S, 0.0, 1.0),
            "info": info,
        }
        return result

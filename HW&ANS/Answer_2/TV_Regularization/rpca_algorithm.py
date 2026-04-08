"""
RPCA 图像修复算法模块 —— 图像读取、ALM 分解与 TV 正则扩展
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
# 基本算子
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


def gradient_x(X: np.ndarray) -> np.ndarray:
    """水平方向前向差分。"""
    return np.roll(X, -1, axis=1) - X


def gradient_y(X: np.ndarray) -> np.ndarray:
    """竖直方向前向差分。"""
    return np.roll(X, -1, axis=0) - X


def divergence(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """差分算子的伴随散度。"""
    return px - np.roll(px, 1, axis=1) + py - np.roll(py, 1, axis=0)


def tv_denoise_primal_dual(
    V: np.ndarray,
    weight: float,
    num_iter: int = 20,
    tau: float = 0.25,
    sigma: float = 0.25,
    theta: float = 1.0,
) -> np.ndarray:
    """
    求解：
        min_X  0.5 ||X - V||_F^2 + weight * TV(X)

    使用原始-对偶迭代实现 TV 近端映射。
    当 weight = 0 时，迭代自然退化为恒等映射，不需要额外分支。
    """
    X = V.copy()
    X_bar = X.copy()
    px = np.zeros_like(V)
    py = np.zeros_like(V)

    for _ in range(num_iter):
        gx = gradient_x(X_bar)
        gy = gradient_y(X_bar)

        px_new = px + sigma * weight * gx
        py_new = py + sigma * weight * gy

        norm = np.maximum(1.0, np.sqrt(px_new ** 2 + py_new ** 2))
        px = px_new / norm
        py = py_new / norm

        X_old = X
        X = (X + tau * (divergence(px, py) + V)) / (1.0 + tau)
        X_bar = X + theta * (X - X_old)

    return X


# ============================================================
# 原始 RPCA（保留作为基线）
# ============================================================

def rpca_alm_channel(
    A: np.ndarray,
    lam: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 500,
    progress_callback=None,
    channel_index: int = 0,
    channel_name: str = "Gray",
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
    """对灰度图或彩图分别执行原始 RPCA。"""
    if A.ndim == 2:
        L, S, info = rpca_alm_channel(
            A,
            lam=lam,
            tol=tol,
            max_iter=max_iter,
            progress_callback=progress_callback,
            channel_index=0,
            channel_name="Gray",
        )
        return L, S, {
            "iterations": info["iterations"],
            "error": info["error"],
            "rank": info["rank"],
            "sparse_ratio": info["sparse_ratio"],
            "lambda": info["lambda"],
            "history": info["history"],
            "channel_infos": [info],
        }

    if A.ndim != 3 or A.shape[2] != 3:
        raise ValueError("Input image must be grayscale or RGB.")

    channel_names = ["R", "G", "B"]
    L_channels = []
    S_channels = []
    channel_infos = []

    for c, name in enumerate(channel_names):
        Lc, Sc, info_c = rpca_alm_channel(
            A[:, :, c],
            lam=lam,
            tol=tol,
            max_iter=max_iter,
            progress_callback=progress_callback,
            channel_index=c,
            channel_name=name,
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
        "history": channel_infos,
        "channel_infos": channel_infos,
    }

    return L, S, info


# ============================================================
# TV-RPCA：在原始 RPCA 的 L 更新后加入 TV 近端步
# ============================================================

def rpca_tv_alm_channel(
    A: np.ndarray,
    lam: float | None = None,
    tv_weight: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 500,
    tv_inner_iter: int = 20,
    progress_callback=None,
    channel_index: int = 0,
    channel_name: str = "Gray",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    使用 ALM + TV 近端映射求解扩展模型：

        min ||L||_* + lambda ||S||_1 + tv_weight * TV(L)
        s.t. A = L + S

    这里的 L 更新分成两步：
      1. 奇异值阈值化，保持低秩结构
      2. TV 近端映射，抑制局部振荡与伪影

    当 tv_weight = 0 时，第二步自然退化为恒等映射，
    整个迭代与原始 RPCA 更新保持一致。
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
        L_half, rank = singular_value_threshold(A - S + Y / mu, 1.0 / mu)
        L = tv_denoise_primal_dual(
            L_half,
            weight=tv_weight / mu,
            num_iter=tv_inner_iter,
        )

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
        "tv_weight": tv_weight,
        "history": history,
        "channel_name": channel_name,
    }

    return L, S, info


def rpca_tv_alm_color(
    A: np.ndarray,
    lam: float | None = None,
    tv_weight: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 500,
    tv_inner_iter: int = 20,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """对灰度图或彩图分别执行 TV-RPCA。"""
    if A.ndim == 2:
        L, S, info = rpca_tv_alm_channel(
            A,
            lam=lam,
            tv_weight=tv_weight,
            tol=tol,
            max_iter=max_iter,
            tv_inner_iter=tv_inner_iter,
            progress_callback=progress_callback,
            channel_index=0,
            channel_name="Gray",
        )
        return L, S, {
            "iterations": info["iterations"],
            "error": info["error"],
            "rank": info["rank"],
            "sparse_ratio": info["sparse_ratio"],
            "lambda": info["lambda"],
            "tv_weight": info["tv_weight"],
            "history": info["history"],
            "channel_infos": [info],
        }

    if A.ndim != 3 or A.shape[2] != 3:
        raise ValueError("Input image must be grayscale or RGB.")

    channel_names = ["R", "G", "B"]
    L_channels = []
    S_channels = []
    channel_infos = []

    for c, name in enumerate(channel_names):
        Lc, Sc, info_c = rpca_tv_alm_channel(
            A[:, :, c],
            lam=lam,
            tv_weight=tv_weight,
            tol=tol,
            max_iter=max_iter,
            tv_inner_iter=tv_inner_iter,
            progress_callback=progress_callback,
            channel_index=c,
            channel_name=name,
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
        "tv_weight": tv_weight,
        "history": channel_infos,
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
        tv_weight: float = 0.0,
        tol: float = 1e-7,
        max_iter: int = 500,
        tv_inner_iter: int = 20,
        progress_callback=None,
    ) -> dict:
        """执行 TV-RPCA 分解。"""
        L, S, info = rpca_tv_alm_color(
            self.original,
            lam=lam,
            tv_weight=tv_weight,
            tol=tol,
            max_iter=max_iter,
            tv_inner_iter=tv_inner_iter,
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
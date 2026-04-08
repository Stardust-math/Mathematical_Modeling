"""
RPCA 图像修复算法模块 —— 掩膜观测、TV 正则与结果封装
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
    image_names: list[str] = []

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
    """散度算子。"""
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
    求解
        min_X 0.5 ||X - V||_F^2 + weight * TV(X)

    当 weight = 0 时，迭代自然退化为恒等映射。
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


def prepare_observation_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """将掩膜统一为二维观测掩膜，1 表示观测到，0 表示已知损坏。"""
    if mask is None:
        return np.ones(shape, dtype=np.float64)

    mask = np.asarray(mask, dtype=np.float64)
    if mask.shape != shape:
        raise ValueError("Mask shape does not match image shape.")

    return np.where(mask > 0.5, 1.0, 0.0)


# ============================================================
# 单通道模型
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
    """原始单通道 RPCA。"""
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


def rpca_tv_masked_channel(
    A: np.ndarray,
    mask: np.ndarray | None = None,
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
    单通道掩膜 TV-RPCA：

        min ||L||_* + lambda ||S||_1 + tv_weight * TV(L)
        s.t. P_Omega(A) = P_Omega(L + S)

    其中 Omega 为观测区域。掩膜为 1 的位置参与约束，掩膜为 0 的
    已知损坏区域不参与数据一致性约束，最后由低秩 + TV 结构自行补全。

    当掩膜全为 1 且 tv_weight = 0 时，该更新自然退化为原始 RPCA。
    """
    A = A.astype(np.float64)
    m, n = A.shape
    M = prepare_observation_mask(mask, (m, n))

    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    A_obs = M * A
    norm_two = np.linalg.norm(A_obs, 2)
    norm_inf = np.max(np.abs(A_obs)) / max(lam, 1e-12)
    dual_norm = max(norm_two, norm_inf, 1e-12)

    Y = A_obs / dual_norm
    L = np.zeros_like(A)
    S = np.zeros_like(A)

    mu = 1.25 / max(norm_two, 1e-12)
    mu_bar = mu * 1e7
    rho = 1.5

    norm_obs = np.linalg.norm(A_obs, ord="fro")
    if norm_obs < 1e-12:
        norm_obs = 1.0

    history = []
    rank = 0

    for it in range(1, max_iter + 1):
        temp_L = M * (A - S + Y / mu) + (1.0 - M) * L
        L_half, rank = singular_value_threshold(temp_L, 1.0 / mu)
        L = tv_denoise_primal_dual(L_half, weight=tv_weight / mu, num_iter=tv_inner_iter)

        temp_S = M * (A - L + Y / mu)
        S = soft_threshold(temp_S, lam / mu)

        residual = M * (A - L - S)
        err = np.linalg.norm(residual, ord="fro") / norm_obs

        Y = Y + mu * residual
        mu = min(mu * rho, mu_bar)

        observed_entries = M > 0.5
        if np.any(observed_entries):
            sparse_ratio = float(np.mean(np.abs(S[observed_entries]) > 1e-3))
        else:
            sparse_ratio = 0.0

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
        "channel_index": channel_index,
        "channel_name": channel_name,
        "observed_ratio": float(np.mean(M)),
    }

    return L, S, info


# ============================================================
# 彩图封装
# ============================================================

def rpca_tv_masked_color(
    A: np.ndarray,
    mask: np.ndarray | None = None,
    lam: float | None = None,
    tv_weight: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 500,
    tv_inner_iter: int = 20,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """对灰度图或彩图分别执行掩膜 TV-RPCA。"""
    if A.ndim == 2:
        L, S, info = rpca_tv_masked_channel(
            A,
            mask=mask,
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
            "observed_ratio": info["observed_ratio"],
            "channel_infos": [info],
        }

    if A.ndim != 3 or A.shape[2] != 3:
        raise ValueError("Input image must be grayscale or RGB.")

    channel_names = ["R", "G", "B"]
    L_channels = []
    S_channels = []
    channel_infos = []

    for c, name in enumerate(channel_names):
        Lc, Sc, info_c = rpca_tv_masked_channel(
            A[:, :, c],
            mask=mask,
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
        "observed_ratio": float(np.mean([item["observed_ratio"] for item in channel_infos])),
        "channel_infos": channel_infos,
    }

    return L, S, info


# ============================================================
# 显示辅助
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


def make_masked_input_display(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """构造用于显示的遮挡输入图。"""
    if image.ndim == 2:
        display = image.copy()
        display[mask < 0.5] = 0.92
        return display

    display = image.copy()
    fill_color = np.array([0.92, 0.92, 0.92], dtype=np.float64)
    display[mask < 0.5] = fill_color
    return display


# ============================================================
# 高层封装
# ============================================================

class ImageRestorer:
    """封装单张图片的读取与修复。"""

    def __init__(self, image_path: str | Path):
        image_path = Path(image_path)
        self.image_path = image_path
        self.name = image_path.name
        self.original = read_image(image_path)

    def default_mask(self) -> np.ndarray:
        """返回全观测掩膜。"""
        h, w = self.original.shape[:2]
        return np.ones((h, w), dtype=np.float64)

    def solve(
        self,
        lam: float | None = None,
        tv_weight: float = 0.0,
        tol: float = 1e-7,
        max_iter: int = 500,
        tv_inner_iter: int = 20,
        mask: np.ndarray | None = None,
        progress_callback=None,
    ) -> dict:
        """执行掩膜 TV-RPCA 分解。"""
        h, w = self.original.shape[:2]
        observation_mask = prepare_observation_mask(mask, (h, w))

        L, S, info = rpca_tv_masked_color(
            self.original,
            mask=observation_mask,
            lam=lam,
            tv_weight=tv_weight,
            tol=tol,
            max_iter=max_iter,
            tv_inner_iter=tv_inner_iter,
            progress_callback=progress_callback,
        )

        result = {
            "original": self.original,
            "masked_input": make_masked_input_display(self.original, observation_mask),
            "input_image": self.original,
            "low_rank": np.clip(L, 0.0, 1.0),
            "sparse": sparse_to_display(S),
            "reconstruction": np.clip(L + S, 0.0, 1.0),
            "mask": observation_mask,
            "info": info,
        }
        return result
### Core Idea

The project studies image restoration through **Robust Principal Component Analysis (RPCA)**.  
Given an observed image matrix $A$, the goal is to decompose it into

- a **low-rank component** $L$, which captures the major image structure, and
- a **sparse component** $S$, which captures noise, corruption, and outliers.

The basic formulation is

$$
\min_{L,S} \|L\|_* + \lambda \|S\|_1
\quad \text{s.t.} \quad
A = L + S.
$$

Here, $\|L\|_*$ is the nuclear norm for encouraging low rank, and $\|S\|_1$ promotes sparsity in the corruption term.

### Progressive Development

This project is built as a sequence of five increasingly richer versions:

- **Basic:** grayscale RPCA decomposition for separating structure and sparse noise
- **Color:** channel-wise extension from grayscale to RGB images
- **GUI_advanced:** improved usability with progress display, interactive inspection, and result export
- **TV_Regularization:** smoother reconstruction by adding a total variation prior
- **Masked:** restoration with manually selected damaged regions excluded from direct fitting

This progressive design makes the project move from a basic matrix decomposition task to a more complete restoration system.

### TV-Regularized Extension

To preserve large structures while reducing local oscillation, the project extends the model by adding a TV term:

$$
\min_{L,S} \|L\|_* + \lambda \|S\|_1 + \gamma\, TV(L)
\quad \text{s.t.} \quad
A = L + S.
$$

Here, $TV(L)$ measures the spatial variation of the recovered image, and $\gamma$ controls the strength of this smoothness prior.

In practice, this version is designed to produce cleaner visual results than plain RPCA, especially when local artifacts remain after decomposition.

### Masked Restoration

For partially known damaged regions, not every pixel should be forced to match the observation.  
Instead, the fidelity constraint is imposed only on the observed region $\Omega$:

$$
\min_{L,S} \|L\|_{*} + \lambda \|S\|_1 + \gamma\, TV(L)
\quad \text{s.t.} \quad
P_{\Omega}(A) = P_{\Omega}(L+S).
$$

Here, $P_{\Omega}(\cdot)$ denotes projection onto the valid observed pixels.  
This allows the damaged area to be ignored during fitting and then completed from the structural prior of the model.

### Interface Improvements

The advanced interface does not change the mathematical model itself, but it makes the restoration process much easier to use and inspect.  
The improved GUI supports features such as interactive viewing, progress feedback, masking operations, and result saving, so the project is not only algorithmic but also demonstrative and user-friendly.

### What This Page Shows

This page presents the project from several complementary angles:

- **Abstract:** a concise summary of the whole homework
- **This overview:** the main idea behind the model progression
- **Image demos:** visual comparison across different versions
- **Video demos:** interface interaction and restoration workflow
- **Poster:** a compact presentation of the final outcome

### Why It Matters

This homework shows how a classical low-rank and sparse decomposition model can be extended step by step into a more practical image restoration framework.  
The final system is not limited to basic denoising: it also supports smoother recovery, richer interaction, and masked completion for manually specified damaged regions.

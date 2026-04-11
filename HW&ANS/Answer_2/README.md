# RPCA Image Restoration Project

## 1. Project Overview

This project implements a progressive image restoration system based on **Robust PCA (RPCA)** and its extensions.  
The project starts from the basic ALM-based RPCA model, then extends to color images, improves the graphical user interface, adds **TV regularization**, and finally supports **masked restoration** for manually selected damaged regions.

The core idea is to decompose an observed image matrix into:

- a **low-rank component** `L`, which captures the main image structure,
- a **sparse component** `S`, which captures corruption, outliers, and damaged regions.

For the masked version, the data consistency constraint is enforced **only on observed pixels**, while the damaged region is automatically completed by the low-rank prior and TV regularization.

---

## 2. Mathematical Models

### 2.1 Basic RPCA

The original RPCA model is:

\[
\min_{L,S} \|L\|_* + \lambda \|S\|_1,
\quad \text{s.t.} \quad A = L + S.
\]

Here:

- \(\|L\|_*\) is the nuclear norm, encouraging low rank,
- \(\|S\|_1\) is the entrywise \(\ell_1\) norm, encouraging sparsity,
- \(A\) is the observed image matrix.

The optimization is solved by the **Augmented Lagrangian Method (ALM)**.

### 2.2 TV-Regularized RPCA

To improve local smoothness and suppress staircase-like artifacts, the project extends the model to:

\[
\min_{L,S} \|L\|_* + \lambda \|S\|_1 + \gamma \, TV(L),
\quad \text{s.t.} \quad A = L + S.
\]

Here:

- \(TV(L)\) is the total variation regularizer,
- \(\gamma\) is the TV regularization weight.

### 2.3 Masked TV-RPCA

For partially known damaged regions, not all pixels need to satisfy the original equality constraint.  
Instead, the model is imposed only on the observed region \(\Omega\):

\[
\min_{L,S} \|L\|_* + \lambda \|S\|_1 + \gamma \, TV(L),
\quad \text{s.t.} \quad P_{\Omega}(A) = P_{\Omega}(L + S).
\]

Here:

- \(P_{\Omega}(\cdot)\) denotes projection onto the observed region,
- damaged pixels are excluded from the data consistency constraint,
- the missing region is completed by the structural prior of the model.

---

## 3. Project Structure

```text
project/
├── Basic/
│   ├── main.py
│   ├── rpca_algorithm.py
│   ├── gui.py
│   └── figs/
│       └──1.png
├── Color/
│   ├── main.py
│   ├── rpca_algorithm.py
│   ├── gui.py
│   └── figs/
│       └──1.png
├── GUI_advanced/
│   ├── main.py
│   ├── rpca_algorithm.py
│   ├── gui.py
│   └── figs/
│       └──1.png
├── TV_Regularization/
│   ├── main.py
│   ├── rpca_algorithm.py
│   ├── gui.py
│   └── figs/
│       └──1.png
├── Masked/
│   ├── main.py
│   ├── rpca_algorithm.py
│   ├── gui.py
│   └── figs/
│       └──1.png
├── README.md
└── environment.yml
```

### File description

- `main.py`  
  Entry point of the project.

- `rpca_algorithm.py`  
  Core algorithm implementation, including ALM-based RPCA, TV regularization, masked restoration, and display-related helper functions.

- `gui.py`  
  Graphical user interface, including image loading, parameter setting, zooming, dragging, masking, saving results, and progress display.

- `figs/`  
  Test images used by the program.

- `environment.yml`  
  Conda environment configuration file.

- `README.md`  
  Project documentation.

---

## 4. Environment Configuration

This project uses Python and a small number of numerical and plotting libraries.

### Recommended environment

- Python 3.12.2
- NumPy
- Matplotlib
- Tk

### Create the environment

Use Conda to create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate rpca-image-restoration
```

---

## 5. How to Run

Each version has its own `main.py` entry file in the corresponding directory.

### Run the Basic version

```bash
cd Basic
python main.py
```

### Run the Color version

```bash
cd Color
python main.py
```

### Run the GUI_advanced version

```bash
cd GUI_advanced
python main.py
```

### Run the TV_Regularization version

```bash
cd TV_Regularization
python main.py
```

### Run the Masked version

```bash
cd Masked
python main.py
```

After launching the GUI, you may:

1. choose or browse an image,
2. adjust the parameters,
3. optionally mark damaged regions using box selection,
4. run the restoration algorithm,
5. save the whole figure or individual panels.

---

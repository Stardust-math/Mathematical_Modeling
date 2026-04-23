## 1. Problem Setting

This project studies planar curve reconstruction from sampled points and compares three complementary viewpoints:

1. **Local interpolation**, represented by cubic Hermite splines and cubic B-spline interpolation.
2. **Global approximation**, represented by polynomial and B-spline least-squares fitting.
3. **Frequency-domain reconstruction**, represented by truncated Fourier series for periodic closed contours.

The goal is not only to obtain visually accurate curves, but also to understand how the reconstruction behavior changes with parameterization, node density, noise, and harmonic truncation.

## 2. Interpolation and Approximation Models

For Hermite interpolation, each interval is represented by a cubic polynomial determined by endpoint values and endpoint derivatives. In matrix form,

$$
\mathbf{r}_i(t)=h_{00}(t)\mathbf{P}_i+h_{10}(t)\mathbf{m}_i+h_{01}(t)\mathbf{P}_{i+1}+h_{11}(t)\mathbf{m}_{i+1},
$$

where $\mathbf{P}_i$ are sampled points, $\mathbf{m}_i$ are estimated tangents, and $h_{00}, h_{10}, h_{01}, h_{11}$ are the standard cubic Hermite basis functions.

For cubic B-spline interpolation, the curve is written as

$$
\mathbf{r}(t)=\sum_{j} \mathbf{C}_j N_{j,3}(t),
$$

where $N_{j,3}(t)$ are cubic B-spline basis functions and $\mathbf{C}_j$ are control points obtained from interpolation constraints.

For global approximation, the polynomial least-squares model minimizes a global residual over the data, while the B-spline least-squares model keeps the same approximation spirit but uses local spline bases. This produces a useful contrast:

- interpolation tends to pass through all nodes and preserve local detail;
- approximation tends to smooth noise and reduce oscillatory instability.

## 3. Evaluation Metrics and Experimental Factors

The main geometric metric is the symmetric Chamfer distance between a dense target set $A$ and sampled fitted points $B$:

$$
d_{\mathrm{ch}}(A,B)=\frac{1}{|A|}\sum_{a\in A}\min_{b\in B}\lVert a-b\rVert_2
+\frac{1}{|B|}\sum_{b\in B}\min_{a\in A}\lVert b-a\rVert_2.
$$

For selected experiments, the report also records average point-to-curve distance, Hausdorff distance, and runtime.

The experiments are organized around four axes:

- **method comparison** on open and closed representative curves;
- **parameterization comparison** for spline interpolation;
- **node-count sensitivity** to study resolution versus accuracy;
- **noise robustness** to compare interpolation with approximation.

The parameterization choice is especially important for nonuniform geometry. Uniform spacing ignores local shape variation, chord-length spacing tracks Euclidean spacing, and Foley-type parameterization further emphasizes turning behavior in highly curved regions.

## 4. Main Experimental Findings

### 4.1 Method comparison

On smooth open curves such as the S-curve and the sinusoidally modulated curve, all four methods can recover the overall shape, but they differ in local behavior. Interpolation methods preserve fine pointwise structure, while least-squares models produce smoother global trends. The local zoom figures make this distinction more visible: small oscillations or local mismatches appear much more clearly after magnification.

### 4.2 Parameterization

The parameterization experiments show that the spline basis alone does not determine the final quality. Even with the same interpolation scheme, different parameter assignments produce visibly different reconstructions. On the more shape-sensitive examples, Foley-style parameterization improves node allocation in high-curvature regions and reduces distortion relative to purely uniform spacing.

### 4.3 Node density

The node-count experiments quantify the expected trade-off: increasing the number of nodes generally lowers the fitting error, but the rate of improvement eventually slows down. For simple shapes, a moderate number of nodes is already sufficient. For more oscillatory contours such as the wavy circle, the error decreases more gradually because the geometry contains higher-frequency components.

### 4.4 Noise robustness

The robustness experiments are one of the clearest contrasts in the report. Under perturbation, interpolation methods inherit the noise more directly because they are constrained to follow all given samples. Least-squares methods, especially polynomial and B-spline approximation, act as implicit smoothers and therefore maintain lower average error when the observations become noisy.

The distribution plots strengthen this conclusion. The boxplot for the S-curve shows a wider and more skewed spread for interpolation-based methods, indicating stronger sensitivity to perturbation and a larger variance across trials. The violin plot for the circle shows that least-squares methods keep a tighter central mass, while interpolation develops longer tails, meaning that the average performance and the failure mode under noise are both less stable.

## 5. Fourier Reconstruction for Closed Curves

For periodic closed contours, the project introduces a Fourier representation based on complex samples $z_n=x_n+i y_n$. After discrete Fourier analysis, the reconstruction with harmonic cutoff $K$ is written as

$$
\hat z(t)=\sum_{k=-K}^{K} c_k e^{ikt},
$$

where $c_k$ are Fourier coefficients computed from the sampled contour.

This representation gives a compact frequency-domain description of the curve:

- small $K$ preserves only coarse global shape;
- moderate $K$ recovers dominant geometric features;
- large $K$ reproduces fine details and sharp local undulations.

The cardioid and wavy-circle examples show this progression clearly. Their static reconstructions visualize how the contour evolves as $K$ increases, and the GIF demonstrations make the epicycle interpretation directly visible. The error-versus-$K$ figures confirm the same trend quantitatively: reconstruction error drops rapidly at first and then enters a saturation regime once the dominant modes have already been captured.

## 6. Website Content Arrangement

Following the report structure, the webpage is arranged in the same logical order:

1. **Abstract**
2. **Project Overview**
3. **Static SVG figures** in the same order as the report figures
4. **GIF demos** for Fourier epicycle reconstruction
5. **Embedded PDF report**

If you later upload `HW_3/Images/GUI.jpeg`, you can also enable the optional GUI screenshot block already reserved in `index.html`.

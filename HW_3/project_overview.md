## 1. Problem Setting

This project studies planar curve reconstruction from sampled points and compares **four fitting models** together with a **Fourier reconstruction module** for periodic closed contours:

1. **Cubic Hermite spline interpolation**
2. **Cubic B-spline interpolation**
3. **Polynomial least-squares fitting**
4. **B-spline least-squares fitting**
5. **Fourier-series reconstruction for periodic closed curves**

The goal is not only to recover visually accurate curves, but also to understand how the reconstruction quality changes with parameterization, node density, noise, and harmonic truncation. The report therefore compares **interpolation versus approximation**, **local basis versus global basis**, and **time-domain fitting versus frequency-domain reconstruction** under a unified evaluation framework.

## 2. Interpolation and Approximation Models

The fitting part of the project is organized into **two interpolation methods** and **two approximation methods**.

### 2.1 Interpolation methods

For **cubic Hermite spline interpolation**, each interval is represented by a cubic polynomial determined by endpoint values and endpoint derivatives:

$$
\mathbf{r}_i(t)=h_{00}(t)\mathbf{P}_i+h_{10}(t)\mathbf{m}_i+h_{01}(t)\mathbf{P}_{i+1}+h_{11}(t)\mathbf{m}_{i+1},
$$

where $\mathbf{P}_i$ are sampled points, $\mathbf{m}_i$ are estimated tangents, and $h_{00}, h_{10}, h_{01}, h_{11}$ are the standard cubic Hermite basis functions.

For **cubic B-spline interpolation**, the curve is written as

$$
\mathbf{r}(t)=\sum_j \mathbf{C}_j N_{j,3}(t),
$$

where $N_{j,3}(t)$ are cubic B-spline basis functions and $\mathbf{C}_j$ are control points determined from interpolation constraints.

Both interpolation methods enforce the sampled data strongly, but they differ in representation: Hermite interpolation is driven directly by point values and tangent information, while B-spline interpolation is expressed through local spline bases and control points.

### 2.2 Approximation methods

For **polynomial least-squares fitting**, the curve coordinates are approximated by global polynomials whose coefficients minimize the residual over all sampled points.

For **B-spline least-squares fitting**, the same least-squares idea is combined with spline basis functions, so the approximation remains smooth while retaining local support.

These two approximation models do not pass exactly through every sample. Instead, they trade interpolation accuracy for smoothing and stability, which becomes especially important in the noisy experiments.

### 2.3 Unified comparison

Taken together, the four fitting models form two natural contrasts:

- **Hermite interpolation vs. B-spline interpolation**: both are interpolatory, but they use different local representations.
- **Polynomial least squares vs. B-spline least squares**: both are approximating, but one is global and the other is spline-based.
- **Interpolation vs. approximation**: interpolation preserves local sample information more aggressively, whereas approximation usually provides stronger smoothing and better robustness.

## 3. Evaluation Metrics and Experimental Factors

The webpage follows the report and records **three geometric distance metrics** together with runtime information.

### 3.1 Symmetric Chamfer distance

The main quantitative metric is the symmetric Chamfer distance between a dense target set $A$ and sampled fitted points $B$:

$$
d_{\mathrm{ch}}(A,B)=\frac{1}{|A|}\sum_{a\in A}\min_{b\in B}\lVert a-b\rVert_2
+\frac{1}{|B|}\sum_{b\in B}\min_{a\in A}\lVert b-a\rVert_2.
$$

It measures the average bidirectional geometric mismatch between the reference curve and the reconstructed curve.

### 3.2 Average point-to-curve distance

The **average point-to-curve distance** measures the mean distance from sampled reference points to the fitted curve. Compared with Chamfer distance, it is more directly tied to average fitting quality from the data side.

### 3.3 Hausdorff distance

The **Hausdorff distance** measures the worst-case geometric deviation between the reference set and the reconstructed set. Compared with the two average-type metrics above, it is more sensitive to local outliers and sharp mismatch.

Therefore, the three metrics serve different purposes:

- **Chamfer distance** captures overall bidirectional average agreement;
- **average point-to-curve distance** emphasizes mean fitting error from the point set to the curve;
- **Hausdorff distance** highlights the largest local deviation.

For selected experiments, the report also records runtime so that geometric accuracy and computational cost can be viewed together.

### 3.4 Experimental factors

The experiments are organized around four axes:

- **method comparison** on representative open and closed curves;
- **parameterization comparison** for spline interpolation;
- **node-count sensitivity** to study sampling resolution versus accuracy;
- **noise robustness** to compare interpolation with approximation.

The parameterization choice is especially important for nonuniform geometry. Uniform spacing ignores local shape variation, chord-length spacing tracks Euclidean spacing, and Foley-type parameterization further emphasizes turning behavior in highly curved regions.

## 4. Main Experimental Findings

### 4.1 Method comparison

On smooth open curves such as the S-curve and the sinusoidally modulated curve, all four fitting methods recover the global geometry, but they behave differently in local detail.

- **Cubic Hermite interpolation** captures local shape sharply through tangent information.
- **Cubic B-spline interpolation** remains interpolatory but usually looks slightly smoother in its piecewise transition.
- **Polynomial least-squares fitting** produces a global approximation and may smooth away local oscillation.
- **B-spline least-squares fitting** combines smoothing with local support and often gives a balanced compromise.

The local zoom figures make these differences much more visible than the global plots.

### 4.2 Parameterization

The parameterization experiments show that the interpolation result depends not only on the spline basis but also on how the parameter values are assigned to the data. Even with the same interpolation model, different parameterizations produce noticeably different reconstructions. On the more shape-sensitive examples, Foley-style parameterization improves node allocation in high-curvature regions and reduces distortion relative to purely uniform spacing.

### 4.3 Node density

The node-count experiments quantify the trade-off between resolution and accuracy. Increasing the number of nodes generally lowers the fitting error, but the improvement eventually slows down. Simple shapes reach saturation quickly, whereas oscillatory shapes such as the wavy circle require more nodes because they contain stronger high-frequency structure.

### 4.4 Noise robustness

The robustness experiments provide the clearest comparison between interpolation and approximation.

- The two **interpolation methods** are more sensitive to perturbation because they are forced to follow the noisy samples directly.
- The two **least-squares approximation methods** are more robust because they smooth the observations while minimizing an overall residual.

The distribution plots reinforce this conclusion. The boxplot for the S-curve shows a wider and more skewed spread for interpolation-based methods, indicating larger variance across trials. The violin plot for the circle shows that least-squares methods keep a tighter central mass, while interpolation develops longer tails. This means the difference is not only in average error, but also in stability and failure mode under noise.

## 5. Fourier Reconstruction for Closed Curves

For periodic closed contours, the project introduces a Fourier representation based on complex samples $z_n=x_n+i y_n$. After discrete Fourier analysis, the reconstruction with harmonic cutoff $K$ is written as

$$
\hat z(t)=\sum_{k=-K}^{K} c_k e^{ikt},
$$

where $c_k$ are Fourier coefficients computed from the sampled contour.

This representation gives a compact frequency-domain description of the curve:

- small $K$ preserves only coarse global shape;
- moderate $K$ recovers dominant geometric features;
- large $K$ reproduces fine details and local undulations.

The cardioid and wavy-circle examples show this progression clearly. Their static reconstructions visualize how the contour evolves as $K$ increases, and the GIF demonstrations make the epicycle interpretation directly visible. The error-versus-$K$ figures confirm the same trend quantitatively: reconstruction error drops rapidly at first and then gradually saturates once the dominant modes have already been captured.

## 6. Website Content Arrangement

Following the report structure, the webpage is arranged in the same logical order:

1. **Abstract**
2. **Project Overview**
3. **Static SVG figures** in the same order as the report figures
4. **GIF demos** for Fourier epicycle reconstruction
5. **Embedded PDF report**

If `HW_3/Images/GUI.jpeg` is uploaded later, the optional GUI screenshot block already reserved in `index.html` can be enabled directly.

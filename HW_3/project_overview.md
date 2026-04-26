## 1. Problem Setting

This project studies planar curve fitting from ordered sampled points under two connected viewpoints:

1. **interpolation**, represented by cubic spline interpolation and cubic B-spline interpolation;
2. **approximation**, represented by polynomial least-squares fitting and B-spline least-squares fitting;
3. **frequency-domain reconstruction**, represented by truncated Fourier series for periodic closed contours.

A planar curve is represented as a parametric mapping

$$
\mathbf r(t)=(x(t),y(t)), \qquad t\in[0,1].
$$

After parameter values are assigned to the sampled points, the two coordinate functions can be fitted separately or through a vector-valued spline basis. The input samples are assumed to be ordered along the underlying curve. If the sample order were unknown, an additional ordering step, such as a traveling-salesman-style path search, would be needed before interpolation or approximation. This ordering problem is not the focus here.

The goal is not only to obtain visually accurate curves, but also to understand how reconstruction quality changes with parameterization, node density, noise, and Fourier harmonic truncation.

## 2. Overall Workflow

<div align="center">
  <img src="./Images/workflow_overview.svg" alt="Overall workflow" style="width:100%; max-width:1400px;">
</div>

The workflow starts from ordered planar samples. For general curve fitting, the samples are first parameterized and then processed by either exact interpolation or least-squares approximation models. For closed contours, the data can also be resampled uniformly along arclength and rewritten as a periodic complex signal. This produces a separate Fourier branch for harmonic truncation and epicycle-style reconstruction. The outputs are compared by geometric distances, runtime, node-count experiments, noise robustness, and visual inspection.

## 3. Parameterization

Suppose the sampled points are

$$
\mathbf p_i=(x_i,y_i), \qquad i=0,1,\dots,n-1.
$$

Before fitting, each point must be assigned a parameter value $t_i$. The implementation compares four rules and then normalizes the resulting parameters to $[0,1]$.

For **uniform parameterization**, the increments are constant:

$$
t_{i+1}-t_i=\mathrm{const}.
$$

For **chord-length parameterization**, the increments are proportional to Euclidean segment length:

$$
t_{i+1}-t_i \propto \|\mathbf p_{i+1}-\mathbf p_i\|_2.
$$

For **centripetal parameterization**, the long-segment effect is weakened by taking a square root:

$$
t_{i+1}-t_i \propto \sqrt{\|\mathbf p_{i+1}-\mathbf p_i\|_2}.
$$

For the curvature-aware **Foley--Nielsen parameterization**, let

$$
d_i=\|\mathbf p_{i+1}-\mathbf p_i\|_2,
\qquad
\widehat\alpha_i=\min\{\pi-\angle(\mathbf p_{i-1},\mathbf p_i,\mathbf p_{i+1}),\pi/2\}.
$$

For interior segments, the increment can be summarized as

$$
t_{i+1}-t_i \propto
 d_i\left(
1+
\frac{3\widehat\alpha_i d_{i-1}}{2d_{i-1}+d_i}
+
\frac{3\widehat\alpha_{i+1} d_i}{2d_i+d_{i+1}}
\right),
$$

with endpoint cases handled by the implementation. In other words, the distance increment is adjusted by local turning information, so regions with sharper geometric change receive more parameter resolution. This is important because the same Euclidean distance may correspond to very different local curvature.

## 4. Interpolation and Approximation Models

### 4.1 Cubic spline interpolation

The cubic spline interpolant fits the two coordinate functions separately with piecewise cubic polynomials:

$$
S_x(t_i)=x_i, \qquad S_y(t_i)=y_i, \qquad i=0,1,\dots,n-1.
$$

The spline preserves $C^2$ continuity across knots, and the final fitted curve is

$$
\mathbf S(t)=(S_x(t),S_y(t)).
$$

Because the interpolation conditions are enforced pointwise, the reconstructed curve passes through every sample point. This makes the method accurate on clean data, but also more sensitive to noisy perturbations.

### 4.2 Cubic B-spline interpolation

The cubic B-spline interpolant represents the curve as

$$
\mathbf C(t)=\sum_{j=0}^{m-1}\mathbf c_j N_{j,p}(t), \qquad p=3,
$$

where $N_{j,p}(t)$ are B-spline basis functions and $\mathbf c_j$ are control points. The control points are solved from the interpolation constraints

$$
\mathbf C(t_i)=\mathbf p_i, \qquad i=0,1,\dots,n-1.
$$

This method is also exactly interpolatory, but it describes the curve through local basis functions and control points rather than through separate piecewise cubic coordinate polynomials.

### 4.3 Polynomial least-squares fitting

The polynomial approximation model uses a degree-$d$ basis:

$$
x(t)\approx\sum_{k=0}^{d}a_k t^k,
\qquad
 y(t)\approx\sum_{k=0}^{d}b_k t^k.
$$

The coefficients are obtained by minimizing

$$
\min_{\mathbf a,\mathbf b}
\sum_{i=0}^{n-1}\left(x_i-\sum_{k=0}^{d}a_k t_i^k\right)^2
+
\sum_{i=0}^{n-1}\left(y_i-\sum_{k=0}^{d}b_k t_i^k\right)^2.
$$

This model does not force the fitted curve to pass through every sample. It trades exact node matching for global smoothing, which can be beneficial when the samples contain noise.

### 4.4 B-spline least-squares fitting

The B-spline approximation model keeps the spline basis but replaces exact interpolation with a least-squares objective:

$$
\min_{\{\mathbf c_j\}}
\sum_{i=0}^{n-1}
\left\|
\mathbf p_i-
\sum_{j=0}^{m-1}\mathbf c_j N_{j,p}(t_i)
\right\|_2^2.
$$

Compared with global polynomial fitting, this model retains local support. Compared with interpolation, it is less sensitive to pointwise perturbations because it is allowed to smooth the samples rather than reproduce every noisy point exactly.

## 5. Quantitative Metrics and Experimental Setting

The main quantitative metric is the symmetric Chamfer distance between a dense target point set $A$ and sampled points on the fitted curve $B$:

$$
d_{\mathrm{ch}}(A,B)=
\frac{1}{|A|}\sum_{a\in A}\min_{b\in B}\|a-b\|_2
+
\frac{1}{|B|}\sum_{b\in B}\min_{a\in A}\|b-a\|_2.
$$

For selected experiments, the report also records the average point-to-curve distance

$$
d_{\mathrm{pc}}(A,\Gamma)=
\frac{1}{|A|}\sum_{a\in A}\min_{x\in\Gamma}\|a-x\|_2,
$$

where $\Gamma$ denotes the fitted curve, and the Hausdorff distance

$$
d_{\mathrm H}(A,B)=
\max\left\{
\sup_{a\in A}\inf_{b\in B}\|a-b\|_2,
\sup_{b\in B}\inf_{a\in A}\|b-a\|_2
\right\}.
$$

Runtime is also recorded as an efficiency measure. Smaller distance values indicate better geometric agreement, while lower runtime indicates lower computational cost.

The experiments use representative open and closed curves, including an S-curve, a sinusoidally modulated open curve, a circle, an ellipse, a cardioid, a cubic polynomial curve, and a wavy circle. Unless otherwise stated, the fitting experiments use 40 sample nodes and evaluate the reconstructed curve on a dense grid of 800 points. The polynomial least-squares model uses degree 7, and the spline models use cubic basis functions. The node-count study uses

$$
n\in\{12,20,30,40,60,80\},
$$

and the noise study adds Gaussian perturbations with

$$
\sigma\in\{0,0.01,0.02,0.04,0.06\}.
$$

Each noise level is repeated over ten random trials. The parameterization comparison is carried out with B-spline interpolation so that the effect of the parameter rule is isolated from the effect of the fitting family.

## 6. Main Experimental Findings

### 6.1 Method comparison

The noiseless experiments show that the best fitting family depends on the target geometry. For smooth low-complexity curves such as the circle and the cubic polynomial curve, polynomial least squares can be very accurate because a low-degree global trend is sufficient. For curves with stronger local bends or repeated oscillations, spline-based methods are more reliable. On the S-curve, cubic spline interpolation gives the smallest error, while on the wavy circle the global polynomial model deteriorates because the oscillatory boundary is difficult to represent with a low-degree polynomial.

The local zoom figures are important because global plots can hide where the error is concentrated. In high-curvature or oscillatory regions, spline methods better preserve local shape, while global polynomial fitting tends to smooth or shift the local geometry.

### 6.2 Parameterization

The parameterization experiments show that parameter assignment is a modeling choice, not a minor preprocessing detail. With the same B-spline interpolation method, different parameter rules produce visibly and quantitatively different reconstructions.

Uniform spacing is generally the weakest choice when the physical geometry changes non-uniformly. Chord-length and Foley--Nielsen parameterization perform better on the cardioid and ellipse, while centripetal parameterization performs best on the sinusoidally modulated open curve. This pattern is consistent with the role of each rule: chord length follows physical distance, centripetal spacing reduces the dominance of long segments, and Foley--Nielsen spacing additionally accounts for local turning.

### 6.3 Node density

As the number of nodes increases, fitting error generally decreases, but the rate of improvement depends on curve complexity. Simple smooth shapes need only a moderate number of nodes. In contrast, oscillatory contours such as the wavy circle require more samples because their geometry contains stronger high-frequency structure. Thus, the node count should match the geometric frequency content of the target rather than being chosen only for visual density.

### 6.4 Noise robustness

Noise robustness creates a clear contrast between interpolation and approximation. Cubic spline interpolation can be highly accurate at zero noise, but its error grows rapidly when sample points are perturbed because every noisy point must still be matched exactly. Least-squares methods relax this exact matching requirement and therefore behave more like smoothers.

The distribution plots reinforce this conclusion. On the S-curve, interpolation-based methods show larger medians, wider spreads, and longer upper tails as the noise level increases. B-spline least squares remains more compact because local smoothing filters part of the perturbation. On the circle, polynomial least squares is especially stable because the target is globally smooth and low-frequency; interpolation instead transfers random radial perturbations into small oscillations along the recovered contour.

## 7. Fourier Reconstruction for Closed Curves

For a closed contour, the most natural model is periodic rather than purely polynomial. After uniform arclength resampling, the code forms the complex signal

$$
z_m=x_m+i y_m, \qquad m=0,1,\dots,M-1.
$$

The discrete Fourier coefficients are

$$
c_k=\frac{1}{M}\sum_{m=0}^{M-1}z_m e^{-i2\pi km/M}, \qquad k\in\mathbb Z,
$$

and the truncated reconstruction is

$$
z_K(t)=\sum_{k=-K}^{K}c_k e^{i2\pi kt}.
$$

This formulation explains the epicycle interpretation: each term $c_k e^{i2\pi kt}$ is a complex vector rotating at frequency $k$, and the full curve is generated by summing circular motions with different radii, phases, and frequencies. Low-order harmonics recover the global outline, while higher-order harmonics restore sharp details and repeated boundary undulations.

The cardioid and wavy-circle examples show this progression clearly. The cardioid is already recognizable at small $K$, but the cusp is rounded until more harmonics are added. The wavy circle improves more slowly because its repeated ripples require higher-frequency coefficients. The error-versus-$K$ curves confirm the same trend quantitatively: reconstruction error drops quickly at first and then enters a saturation regime once the dominant modes have been captured.

## 8. Interactive Interface

Beyond the batch experiments, the project provides a Streamlit interface for interactive inspection. The left control panel allows the user to select the curve family, fitting method, number of nodes, parameterization rule, and Fourier order. The main panel displays the reconstructed curve, sampled points, and corresponding quantitative summary. This interface is useful for checking how small changes in modeling choices affect the geometry, especially when the difference is difficult to judge from CSV tables alone.

## 9. Takeaways

The project supports four main conclusions. First, the fitting method should match the target geometry: global polynomial least squares is effective for smooth low-complexity curves, while spline-based methods are more reliable for local bends and oscillatory structures. Second, exact interpolation is valuable on clean data but becomes sensitive under noise. Third, parameterization strongly affects interpolation quality and should be treated as part of the model design. Finally, closed curves are naturally periodic objects, and Fourier reconstruction provides both an accurate and visually interpretable way to analyze them through harmonic components.

## 1. Problem Setting

This project studies planar curve reconstruction from sampled points and compares three complementary viewpoints:

1. **local interpolation**, represented by cubic Hermite splines and cubic B-spline interpolation;
2. **global approximation**, represented by polynomial and B-spline least-squares fitting;
3. **frequency-domain reconstruction**, represented by truncated Fourier series for periodic closed contours.

The goal is not only to obtain visually accurate curves, but also to understand how reconstruction changes with parameterization, node density, noise, and harmonic truncation.

## 2. Overall Workflow

<div align="center">
  <img src="./Images/workflow_overview.svg" alt="Overall workflow" style="width:100%; max-width:1400px;">
</div>

The workflow starts from scattered planar samples, assigns parameters, applies either interpolation or approximation models, evaluates the reconstructed curves quantitatively, and finally visualizes the results on the webpage. For periodic closed contours, a Fourier branch is added to study harmonic truncation and epicycle-style reconstruction.

## 3. Parameterization

The parameter value attached to each sample point is crucial for all spline-based models. The report compares several standard strategies.

For **uniform parameterization**,

$$
t_i = \frac{i}{n}, \qquad i=0,1,\dots,n.
$$

For **chord-length parameterization**,

$$
t_0=0, \qquad
t_i = t_{i-1} + \|P_i - P_{i-1}\|, \qquad i=1,2,\dots,n,
$$

followed by normalization to $[0,1]$.

For the curvature-aware **Foley-type parameterization**, the increment is adjusted by local turning information. In the implementation, the distance increment is modified by the turning angle and then averaged between adjacent segments so that parameter spacing becomes more sensitive to regions with rapid directional change.

## 4. Interpolation and Approximation Models

### 4.1 Cubic Hermite spline interpolation

On each interval $[t_i,t_{i+1}]$, the curve is written in the same form as in the report:

$$
C_i(t)=h_{00}(u)P_i+h_{10}(u)(t_{i+1}-t_i)M_i+h_{01}(u)P_{i+1}+h_{11}(u)(t_{i+1}-t_i)M_{i+1},
$$

where

$$
u=\frac{t-t_i}{t_{i+1}-t_i},
$$

and the cubic basis functions are

$$
h_{00}(u)=2u^3-3u^2+1, \qquad h_{10}(u)=u^3-2u^2+u,
$$
$$
h_{01}(u)=-2u^3+3u^2, \qquad h_{11}(u)=u^3-u^2.
$$

The tangent vectors are estimated by finite differences:

$$
M_i = \frac{P_{i+1}-P_{i-1}}{t_{i+1}-t_{i-1}}, \qquad i=1,\dots,n-1,
$$

with one-sided endpoint formulas

$$
M_0 = \frac{P_1-P_0}{t_1-t_0},
\qquad
M_n = \frac{P_n-P_{n-1}}{t_n-t_{n-1}}.
$$

This method is fully interpolatory and preserves local geometric detail, but it is also more sensitive to noisy samples.

### 4.2 Cubic B-spline interpolation

The cubic B-spline interpolation model first constructs a knot vector

$$
U=(u_0,u_1,\dots,u_{n+4}),
$$

with boundary repetition

$$
u_0=u_1=u_2=u_3=0, \qquad
u_{n+1}=u_{n+2}=u_{n+3}=u_{n+4}=1,
$$

and interior knots determined from the parameter values:

$$
u_{j+3}=\frac{t_j-t_0}{t_n-t_0}, \qquad j=1,2,\dots,n-1.
$$

The interpolation constraints are written as

$$
D_i = \frac{1}{6}C_{i-1}+\frac{2}{3}C_i+\frac{1}{6}C_{i+1},
\qquad i=1,2,\dots,n-1,
$$

with endpoint conditions

$$
D_0=C_0, \qquad D_n=C_n.
$$

The final curve is

$$
C(t)=\sum_{j=0}^{n} C_j N_{j,3}(t),
$$

where $N_{j,3}(t)$ are cubic B-spline basis functions. Compared with Hermite interpolation, this representation has stronger smoothness and a more control-point-oriented structure.

### 4.3 Polynomial least-squares fitting

For the polynomial approximation model, the report uses the least-squares objective

$$
\min_{a,b} \sum_{i=0}^{n}
\left[x_i-\sum_{k=0}^{m} a_k\phi_k(t_i)\right]^2
+
\left[y_i-\sum_{k=0}^{m} b_k\phi_k(t_i)\right]^2,
$$

with basis functions

$$
\phi_k(t)=t^k.
$$

The fitted coordinates are

$$
x(t)=\sum_{k=0}^{m} a_k\phi_k(t),
\qquad
y(t)=\sum_{k=0}^{m} b_k\phi_k(t),
$$

and the normal equations are

$$
A^\top A a = A^\top x,
\qquad
A^\top A b = A^\top y,
$$

where

$$
A_{ik}=\phi_k(t_i).
$$

This model is global and smoothing by nature, so it often behaves better than exact interpolation when noise is present.

### 4.4 B-spline least-squares fitting

The spline approximation version minimizes

$$
\min_{C_j}
\sum_{i=0}^{n}
\left\|Q_i-\sum_j C_j N_{j,p}(t_i)\right\|^2,
$$

which leads to the normal equations

$$
\mathbf{B}^\top \mathbf{B}\mathbf{C}=\mathbf{B}^\top \mathbf{Q}.
$$

The fitted curve keeps the spline form

$$
C(t)=\sum_j C_j N_{j,p}(t).
$$

This method combines the smoothing effect of approximation with the local support and geometric flexibility of spline bases, which is why it performs especially well in the noisy experiments.

## 5. Quantitative Metrics

The report uses several geometric distances to compare a dense target point set $A$ and sampled points on the fitted curve $B$.

### 5.1 Symmetric Chamfer distance

The main metric is the symmetric Chamfer distance:

$$
d_{\mathrm{ch}}(A,B)=\frac{1}{|A|}\sum_{a\in A}\min_{b\in B}\lVert a-b\rVert_2
+\frac{1}{|B|}\sum_{b\in B}\min_{a\in A}\lVert b-a\rVert_2.
$$

A smaller value indicates a better overall geometric match.

### 5.2 Average point-to-curve distance

For selected experiments, the report also uses the average point-to-curve distance:

$$
d_{\mathrm{avg}}(A,B)=\frac{1}{|A|}\sum_{a\in A}\min_{b\in B}\lVert a-b\rVert_2.
$$

This quantity emphasizes the average deviation from the reference curve to the fitted curve.

### 5.3 Hausdorff distance

To measure worst-case geometric discrepancy, the Hausdorff distance is defined as

$$
d_{\mathrm{H}}(A,B)=\max\Bigg\{\max_{a\in A}\min_{b\in B}\lVert a-b\rVert_2,
\max_{b\in B}\min_{a\in A}\lVert b-a\rVert_2\Bigg\}.
$$

Unlike Chamfer distance, this metric is dominated by the largest local mismatch.

In addition, the experiments record runtime and direct visual comparison.

## 6. Main Experimental Findings

### 6.1 Method comparison

On smooth open curves such as the S-curve and the sinusoidally modulated curve, all four methods recover the overall shape, but they differ in local behavior. Interpolation methods preserve pointwise structure more faithfully, while least-squares models produce smoother global trends. The zoomed-in figures make these differences much more visible.

### 6.2 Parameterization

The parameterization experiments show that the spline basis alone does not determine final quality. Even with the same interpolation scheme, different parameter assignments produce visibly different reconstructions. On more shape-sensitive examples, the Foley-style strategy allocates parameters more effectively in high-curvature regions and reduces distortion relative to purely uniform spacing.

### 6.3 Node density

As the number of nodes increases, fitting error generally decreases, but the rate of improvement eventually slows down. Simple shapes need only a moderate node count, while oscillatory contours such as the wavy circle require more samples because their geometry contains stronger high-frequency content.

### 6.4 Noise robustness

The robustness experiments create one of the clearest contrasts in the project. Interpolation methods inherit perturbations more directly because they are constrained to pass through the samples. Least-squares methods, especially polynomial and B-spline approximation, act as implicit smoothers and therefore maintain lower average error under noise.

The distribution plots reinforce this conclusion. The boxplot for the S-curve shows a wider and more skewed spread for interpolation-based methods, indicating stronger sensitivity to perturbation and larger variance across trials. The violin plot for the circle shows that least-squares methods keep a tighter central mass, while interpolation develops longer tails, meaning that both average performance and failure mode are less stable.

## 7. Fourier Reconstruction for Closed Curves

For periodic closed contours, the project introduces a Fourier representation based on complex samples $z_n=x_n+i y_n$. After discrete Fourier analysis, the reconstruction with harmonic cutoff $K$ is written as

$$
\hat z(t)=\sum_{k=-K}^{K} c_k e^{ikt},
$$

where $c_k$ are Fourier coefficients computed from the sampled contour.

This gives a compact frequency-domain description of the curve:

- small $K$ preserves only coarse global shape;
- moderate $K$ recovers dominant geometric features;
- large $K$ reproduces fine details and sharp local undulations.

The cardioid and wavy-circle examples show this progression clearly. Their static reconstructions visualize how the contour evolves as $K$ increases, and the GIF demonstrations make the epicycle interpretation directly visible. The error-versus-$K$ figures confirm the same trend quantitatively: reconstruction error drops rapidly at first and then enters a saturation regime once the dominant modes have already been captured.

## 8. Website Content Arrangement

Following the report structure, the webpage is arranged in the same logical order:

1. **Abstract**
2. **Project overview and workflow**
3. **Static SVG figures** in the same order as the report figures
4. **GIF demos** for Fourier epicycle reconstruction
5. **Embedded PDF report**

If you later upload `HW_3/Images/GUI.jpeg`, you can also enable the optional GUI screenshot block already reserved in `index.html`.

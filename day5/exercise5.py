# Exercise 1: SciPy — Linear Algebra & Statistics
# ==================================================


import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, stats


def section(title):
    print(f"\n{'='*55}")
    print(f" {title}")
    print('='*55)


# ============================================================
# LINEAR ALGEBRA
# ============================================================

section("1a-b: Define A and b")

A = np.array([[1, -2, 3],
              [4,  5, 6],
              [7,  1, 9]], dtype=float)

b = np.array([1, 2, 3], dtype=float)

print("A =\n", A)
print("b =", b)


# ------------------------------------------------------------
section("1c: Solve A x = b")

x = linalg.solve(A, b)
print("Solution x =", x)


# ------------------------------------------------------------
section("1d: Verify solution A @ x == b")

residual = A @ x - b
print("A @ x       =", A @ x)
print("b           =", b)
print("Residual    =", residual)
print("Max |resid| =", np.max(np.abs(residual)))
print("Correct     :", np.allclose(A @ x, b))


# ------------------------------------------------------------
section("1e: Random 3x3 matrix B — solve A X = B")

rng = np.random.default_rng(42)
B = rng.integers(1, 10, size=(3, 3)).astype(float)
print("B =\n", B)

X = linalg.solve(A, B)
print("Solution X =\n", X)
print("Verify A @ X == B:", np.allclose(A @ X, B))


# ------------------------------------------------------------
section("1f: Eigenvalue problem for A")

eigenvalues, eigenvectors = linalg.eig(A)
print("Eigenvalues  :", eigenvalues)
print("Eigenvectors (columns):\n", eigenvectors)

# Verify: A @ v == lambda * v  for each eigenpair
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v   = eigenvectors[:, i]
    ok  = np.allclose(A @ v, lam * v)
    print(f"  Eigenpair {i}: A@v == λv → {ok}")


# ------------------------------------------------------------
section("1g: Inverse and determinant of A")

A_inv = linalg.inv(A)
det_A = linalg.det(A)

print("Inverse of A:\n", A_inv)
print("Determinant of A:", det_A)
print("Verify A @ A_inv == I:", np.allclose(A @ A_inv, np.eye(3)))


# ------------------------------------------------------------
section("1h: Matrix norms of A")

for order in [1, 2, np.inf, 'fro']:
    print(f"  norm(A, ord={str(order):>4}) = {linalg.norm(A, ord=order):.6f}")


# ============================================================
# STATISTICS
# ============================================================

section("2a: Poisson distribution — PMF, CDF, histogram")

mu = 4.0
poisson = stats.poisson(mu)

k = np.arange(0, 15)
pmf = poisson.pmf(k)
cdf = poisson.cdf(k)
samples_p = poisson.rvs(size=1000, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(f"Poisson distribution (μ={mu})", fontsize=12)

axes[0].bar(k, pmf, color='#2196F3', alpha=0.8, width=0.6)
axes[0].set_title("PMF")
axes[0].set_xlabel("k"); axes[0].set_ylabel("P(X=k)")
axes[0].grid(alpha=0.3, ls='--')

axes[1].step(k, cdf, where='post', color='#2196F3', lw=2)
axes[1].set_title("CDF")
axes[1].set_xlabel("k"); axes[1].set_ylabel("P(X≤k)")
axes[1].grid(alpha=0.3, ls='--')

axes[2].hist(samples_p, bins=range(0, 15), color='#2196F3',
             alpha=0.8, edgecolor='white', density=True)
axes[2].bar(k, pmf, color='red', alpha=0.4, width=0.6, label='PMF')
axes[2].set_title("1000 samples vs PMF")
axes[2].set_xlabel("k"); axes[2].legend()
axes[2].grid(alpha=0.3, ls='--')

plt.tight_layout()
plt.savefig("poisson.png", dpi=130, bbox_inches='tight')
plt.show()
print("Saved: poisson.png")


# ------------------------------------------------------------
section("2b: Normal distribution — PDF, CDF, histogram")

loc, scale = 0.0, 1.0
normal = stats.norm(loc=loc, scale=scale)

x_cont = np.linspace(-4, 4, 300)
pdf = normal.pdf(x_cont)
cdf_n = normal.cdf(x_cont)
samples_n = normal.rvs(size=1000, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(f"Normal distribution (μ={loc}, σ={scale})", fontsize=12)

axes[0].plot(x_cont, pdf, color='#FF5722', lw=2)
axes[0].fill_between(x_cont, pdf, alpha=0.2, color='#FF5722')
axes[0].set_title("PDF"); axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].grid(alpha=0.3, ls='--')

axes[1].plot(x_cont, cdf_n, color='#FF5722', lw=2)
axes[1].set_title("CDF"); axes[1].set_xlabel("x"); axes[1].set_ylabel("F(x)")
axes[1].grid(alpha=0.3, ls='--')

axes[2].hist(samples_n, bins=30, color='#FF5722',
             alpha=0.7, edgecolor='white', density=True)
axes[2].plot(x_cont, pdf, color='black', lw=1.5, label='PDF')
axes[2].set_title("1000 samples vs PDF")
axes[2].set_xlabel("x"); axes[2].legend()
axes[2].grid(alpha=0.3, ls='--')

plt.tight_layout()
plt.savefig("normal.png", dpi=130, bbox_inches='tight')
plt.show()
print("Saved: normal.png")


# ------------------------------------------------------------
section("2c: Two-sample t-test — same distribution?")

rng = np.random.default_rng(0)

# Same distribution (both N(0,1)) → expect p > 0.05
s1 = rng.normal(loc=0, scale=1, size=100)
s2 = rng.normal(loc=0, scale=1, size=100)

# Different distributions (N(0,1) vs N(1,1)) → expect p < 0.05
s3 = rng.normal(loc=1, scale=1, size=100)

t12, p12 = stats.ttest_ind(s1, s2)
t13, p13 = stats.ttest_ind(s1, s3)

print("\nTest: s1 vs s2  (both N(0,1) — should be same)")
print(f"  t = {t12:.4f},  p = {p12:.4f}")
print(f"  Same distribution? {'YES (p > 0.05)' if p12 > 0.05 else 'NO (p ≤ 0.05)'}")

print("\nTest: s1 vs s3  (N(0,1) vs N(1,1) — should differ)")
print(f"  t = {t13:.4f},  p = {p13:.4f}")
print(f"  Same distribution? {'YES (p > 0.05)' if p13 > 0.05 else 'NO (p ≤ 0.05)'}")
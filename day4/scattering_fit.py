import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


def load_dataset(path):
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} must be a 2D array with at least two columns [q, I(q)].")

    q = data[:, 0].astype(float)
    intensity = data[:, 1].astype(float)

    finite_mask = np.isfinite(q) & np.isfinite(intensity)
    q = q[finite_mask]
    intensity = intensity[finite_mask]

    if q.size < 2:
        raise ValueError(f"{path} has fewer than 2 finite points after cleaning.")

    order = np.argsort(q)
    q = q[order]
    intensity = intensity[order]

    return q, intensity


def fit_scale(q_exp, i_exp, q_model, i_model):
    q_min = max(np.min(q_exp), np.min(q_model))
    q_max = min(np.max(q_exp), np.max(q_model))

    if q_min >= q_max:
        raise ValueError("Experimental and model q ranges do not overlap.")

    overlap_mask = (q_exp >= q_min) & (q_exp <= q_max)
    q_fit = q_exp[overlap_mask]
    i_exp_fit = i_exp[overlap_mask]

    if q_fit.size < 2:
        raise ValueError("Not enough experimental points in overlap range for fitting.")

    # Restrict interpolation to overlap domain only (no extrapolation).
    model_interp = interp1d(
        q_model,
        i_model,
        kind="linear",
        bounds_error=True,
        assume_sorted=True,
    )

    i_model_fit = model_interp(q_fit)

    def objective(scale):
        residuals = i_exp_fit - scale * i_model_fit
        return np.sum(residuals ** 2)

    result = minimize_scalar(objective, bounds=(1e-8, 1e3), method="bounded")

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    best_scale = result.x
    residuals = i_exp_fit - best_scale * i_model_fit
    sse = np.sum(residuals ** 2)
    rmse = np.sqrt(np.mean(residuals ** 2))

    ss_tot = np.sum((i_exp_fit - np.mean(i_exp_fit)) ** 2)
    r_squared = np.nan if ss_tot == 0 else 1.0 - (sse / ss_tot)

    return {
        "scale": best_scale,
        "q_fit": q_fit,
        "i_exp_fit": i_exp_fit,
        "i_model_fit": i_model_fit,
        "residuals": residuals,
        "sse": sse,
        "rmse": rmse,
        "r_squared": r_squared,
        "q_min": q_min,
        "q_max": q_max,
        "n_points": int(q_fit.size),
        "opt_result": result,
    }


def plot_fit(fit_result, output_path="ipa_fit.png"):
    q_fit = fit_result["q_fit"]
    i_exp_fit = fit_result["i_exp_fit"]
    i_model_scaled = fit_result["scale"] * fit_result["i_model_fit"]
    residuals = fit_result["residuals"]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.scatter(q_fit, i_exp_fit, s=20, alpha=0.8, label="Experimental", color="#1f77b4")
    ax1.plot(q_fit, i_model_scaled, lw=2.0, label="Scaled model", color="#d62728")
    ax1.set_ylabel("Scattering strength I(q)")
    ax1.set_title("Experimental vs. scaled theoretical model")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend()

    ax2.axhline(0.0, color="black", lw=1.0)
    ax2.scatter(q_fit, residuals, s=18, alpha=0.8, color="#2ca02c")
    ax2.set_xlabel("Scattering vector q")
    ax2.set_ylabel("Residual")
    ax2.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    q_exp, i_exp = load_dataset("I_q_IPA_exp.npy")
    q_model, i_model = load_dataset("I_q_IPA_model.npy")

    fit = fit_scale(q_exp, i_exp, q_model, i_model)
    plot_fit(fit, output_path="ipa_fit.png")

    opt = fit["opt_result"]
    print("Fit complete")
    print(f"Overlap q-range: [{fit['q_min']:.6g}, {fit['q_max']:.6g}]")
    print(f"Points used: {fit['n_points']}")
    print(f"Best scale factor: {fit['scale']:.8g}")
    print(f"SSE: {fit['sse']:.8g}")
    print(f"RMSE: {fit['rmse']:.8g}")
    print(f"R^2: {fit['r_squared']:.8g}")
    print(f"Optimizer success: {opt.success}, iterations: {opt.nfev}")
    print("Saved figure: ipa_fit.png")


if __name__ == "__main__":
    main()

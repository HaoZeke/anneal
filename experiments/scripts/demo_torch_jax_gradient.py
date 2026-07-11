"""Ceres-style gradient plug-in demo for `anneal.run_hmc`.

The Rust HMC sampler accepts ANY Python callable returning the
gradient ndarray. This script shows three plug-in patterns:

  (1) hand-coded analytic gradient (fastest)
  (2) jax.grad(f) (auto-differentiation; trivial wrapper)
  (3) torch autograd via .backward() (auto-differentiation; one-liner wrapper)

All three drive the same `anneal.run_hmc` function -- the SA core does
not know which framework produced the gradient.
"""

from __future__ import annotations

import numpy as np

from anneal import run_hmc


# ---- The objective: Rosenbrock 5D ----------------------------------------


def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


# ---- (1) Analytic gradient -----------------------------------------------


def rosenbrock_grad_analytic(x):
    n = len(x)
    g = np.zeros(n, dtype=np.float64)
    for i in range(n - 1):
        g[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
        g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    return g


# ---- (2) JAX wrapper -- import jax only if available ----------------------


def make_jax_gradient():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        return None

    def f(x):
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    grad_f = jax.grad(f)
    # JAX returns a DeviceArray; convert to numpy at the boundary.
    return lambda x: np.asarray(grad_f(x), dtype=np.float64)


# ---- (3) torch wrapper -- import torch only if available ------------------


def make_torch_gradient():
    try:
        import torch
    except ImportError:
        return None

    def grad(x):
        t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        loss = torch.sum(100.0 * (t[1:] - t[:-1] ** 2) ** 2 + (1.0 - t[:-1]) ** 2)
        loss.backward()
        return t.grad.detach().numpy().astype(np.float64)

    return grad


# ---- Run all three through run_hmc ----------------------------------------


def main():
    low = np.full(5, -2.048)
    high = np.full(5, 2.048)
    options = dict(
        t_init=5.0, epsilon=0.05, l_steps=5, n_epochs=30, steps_per_epoch=50, seed=42
    )

    print("Rosenbrock 5D, three gradient providers, anneal.run_hmc driver")
    print()

    # (1) analytic
    h = run_hmc(rosenbrock, rosenbrock_grad_analytic, low, high, **options)
    print(
        f"  analytic gradient: best_val = {h.best_val:.6f}  pos[0] = {h.best_pos[0]:.4f}"
    )

    # (2) jax
    jax_grad = make_jax_gradient()
    if jax_grad is not None:
        h = run_hmc(rosenbrock, jax_grad, low, high, **options)
        print(
            f"  jax.grad:          best_val = {h.best_val:.6f}  pos[0] = {h.best_pos[0]:.4f}"
        )
    else:
        print("  jax.grad:          (jax not installed; skipped)")

    # (3) torch
    torch_grad = make_torch_gradient()
    if torch_grad is not None:
        h = run_hmc(rosenbrock, torch_grad, low, high, **options)
        print(
            f"  torch.autograd:    best_val = {h.best_val:.6f}  pos[0] = {h.best_pos[0]:.4f}"
        )
    else:
        print("  torch.autograd:    (torch not installed; skipped)")

    print()
    print("All three drivers go through the same Rust HMC kernel via")
    print("anneal._core.run_hmc; the gradient provider is opaque to the")
    print("SA core. Plug in whatever your project already uses.")


if __name__ == "__main__":
    main()

"""XC Python bindings against scalar functional derivatives and array contracts."""

import numpy as np
import pytest

import rgpot

pytestmark = pytest.mark.skipif(
    not rgpot.has_xckernel, reason="built without -Dwith_xckernel"
)


def fixture():
    chi = np.array([[0.5, 0.25, -0.5], [0.25, -0.5, 0.75]])
    dchi = np.array(
        [
            [[0.25, 0.5, -0.25], [-0.5, 0.25, 0.5]],
            [[0.5, -0.25, 0.0], [0.25, 0.5, -0.5]],
            [[0.0, 0.25, 0.5], [0.5, 0.0, 0.25]],
        ]
    )
    lapl = np.array([[0.25, -0.5, 0.75], [-0.25, 0.5, 0.25]])
    weights = np.array([0.5, 0.25, 0.75])
    density = np.array([[0.5, 0.25], [-0.125, 0.75]])
    return chi, dchi, lapl, weights, density


def fields(chi, dchi, lapl, density):
    rho = np.einsum("ug,uv,vg->g", chi, density, chi)
    grad = np.einsum("aug,uv,vg->ag", dchi, density, chi) + np.einsum(
        "ug,uv,avg->ag", chi, density, dchi
    )
    lapl_rho = (
        np.einsum("ug,uv,vg->g", lapl, density, chi)
        + np.einsum("ug,uv,vg->g", chi, density, lapl)
        + 2 * np.einsum("aug,uv,avg->g", dchi, density, dchi)
    )
    return rho, grad, lapl_rho


def test_required_kernel_families():
    assert rgpot.has_xckernel
    assert len(rgpot.XcKernel.catalog()) == 32
    for suffix in (
        "r_o1",
        "ua_o1",
        "ub_o1",
        "r_o2",
        "ua_o2",
        "ub_o2",
        "st_o2_p",
        "st_o2_m",
    ):
        assert f"xck_mgga_lapl_{suffix}" in rgpot.XcKernel.catalog()


@pytest.mark.parametrize("column_scalars", [False, True])
def test_lda_needs_only_basis_values(column_scalars):
    chi, _, _, weights, _ = fixture()
    vrho = np.array([0.25, -0.5, 0.75])
    scal = {"w": weights, "vrho": vrho}
    if column_scalars:
        scal = {name: values[:, None] for name, values in scal.items()}
    actual = rgpot.XcKernel("xck_lda_r_o1").contract(chi, None, scal)
    expected = (chi * (weights * vrho)) @ chi.T
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_laplacian_fock_is_the_scalar_energy_derivative():
    chi, dchi, lapl, weights, density = fixture()
    rho, grad, lapl_rho = fields(chi, dchi, lapl, density)
    scal = {
        "w": weights,
        "vrho": rho + lapl_rho / 8,
        "vsigma": np.full(3, 0.25),
        "vlapl": 1.5 * lapl_rho + rho / 8,
        **{f"grad_rho_{axis}": grad[i] for i, axis in enumerate("xyz")},
    }
    kernel = rgpot.XcKernel("xck_mgga_lapl_r_o1")
    actual = kernel.contract(chi, dchi, scal, lapl_chi=lapl)

    def energy(matrix):
        r, g, l = fields(chi, dchi, lapl, matrix)
        return weights @ (
            r * r / 2 + np.sum(g * g, axis=0) / 4 + 0.75 * l * l + r * l / 8
        )

    expected = np.empty_like(density)
    step = 1 / 1024
    for index in np.ndindex(density.shape):
        plus, minus = density.copy(), density.copy()
        plus[index] += step
        minus[index] -= step
        expected[index] = (energy(plus) - energy(minus)) / (2 * step)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="rc=2"):
        kernel.contract(chi, dchi, scal)


def test_density_fields_match_cartesian_contractions():
    chi, dchi, lapl, _, density = fixture()
    actual = rgpot.XcKernel.fields_from_density(chi, dchi, density, lapl_chi=lapl)
    rho, grad, lapl_rho = fields(chi, dchi, lapl, density)
    expected = {
        "rho": rho[:, None],
        "grad_rho": grad,
        "sigma": np.sum(grad * grad, axis=0)[:, None],
        "lapl": lapl_rho[:, None],
        "tau": (np.einsum("aug,uv,avg->g", dchi, density, dchi) / 2)[:, None],
    }
    for name, values in expected.items():
        np.testing.assert_allclose(actual[name], values, rtol=0, atol=0)


# Exact text from grid_from_numpy / scal_from_dict in the binding.
# The symbolic ranks are literals, so the match is anchored on them.
_OPERAND_SHAPE_MESSAGE = {
    "dchi": r"^dchi: expected shape \(3, nbf, npts\)$",
    "lapl_chi": r"^lapl_chi: expected shape \(nbf, npts\)$",
    "hess_chi": r"^hess_chi: expected shape \(6, nbf, npts\)$",
}


@pytest.mark.parametrize(
    "operand,shape",
    [("dchi", (2, 3, 3)), ("lapl_chi", (3, 2)), ("hess_chi", (2, 6, 3))],
)
@pytest.mark.parametrize("operation", ["contract", "fields_from_density"])
def test_rejects_wrong_operand_shapes(operand, shape, operation):
    chi, dchi, lapl, weights, density = fixture()
    operands = {"dchi": dchi, "lapl_chi": lapl}
    operands[operand] = np.zeros(shape)
    with pytest.raises(ValueError, match=_OPERAND_SHAPE_MESSAGE[operand]):
        if operation == "contract":
            rgpot.XcKernel("xck_lda_r_o1").contract(
                chi, scal={"w": weights, "vrho": weights}, **operands
            )
        else:
            rgpot.XcKernel.fields_from_density(chi, density=density, **operands)


@pytest.mark.parametrize("shape", [(), (2,), (3, 2), (1, 3), (3, 1, 1)])
def test_rejects_wrong_scalar_shapes(shape):
    chi, _, _, weights, _ = fixture()
    with pytest.raises(
        ValueError,
        match=r"^scalar vrho: expected shape \(npts,\) or \(npts, 1\)$",
    ):
        rgpot.XcKernel("xck_lda_r_o1").contract(
            chi, None, {"w": weights, "vrho": np.ones(shape)}
        )

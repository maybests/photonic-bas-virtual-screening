import numpy as np
import matplotlib.pyplot as plt
from skfem import *


def run_3d_fem_thermal(k_sub):
    Lx, Ly = 250e-6, 150e-6

    z_sub = np.linspace(0, 50e-6, 12)
    z_kap = np.linspace(50e-6, 50.1e-6, 4)[1:]
    z_ox = np.linspace(50.1e-6, 50.4e-6, 6)[1:]
    z_wg = np.linspace(50.4e-6, 50.6e-6, 4)[1:]
    Z = np.concatenate([z_sub, z_kap, z_ox, z_wg])

    X = np.linspace(0, Lx, 35)
    Y = np.linspace(0, Ly, 30)

    mesh = MeshHex().init_tensor(X, Y, Z)
    basis = Basis(mesh, ElementHex1())

    R_kapitza = 1e-8
    d_kap = 0.1e-6
    k_kap = d_kap / R_kapitza
    k_ox = 1.4
    k_wg = 148.0

    P_heater = 15e-3
    V_heater = 200e-6 * 15e-6 * 0.2e-6
    q_vol = P_heater / V_heater

    @BilinearForm
    def conduction(u, v, w):
        z = w.x[2]
        k = np.where(z <= 50e-6, k_sub,
            np.where(z <= 50.1e-6, k_kap,
            np.where(z <= 50.4e-6, k_ox, k_wg)))
        return k * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1] + u.grad[2] * v.grad[2])

    @LinearForm
    def heat_source(v, w):
        x, y, z = w.x[0], w.x[1], w.x[2]
        in_heater = (x <= 100e-6) & (y <= 7.5e-6) & (z > 50.4e-6)
        return np.where(in_heater, q_vol * v, 0.0)

    A = asm(conduction, basis)
    b = asm(heat_source, basis)

    bottom_dofs = basis.get_dofs(lambda x: np.isclose(x[2], 0.0))
    A_bc, b_bc = enforce(A, b, D=bottom_dofs)

    dT_fem = solve(A_bc, b_bc)

    coords = basis.doflocs
    wg_mask = (coords[0] <= 100e-6) & (coords[1] <= 7.5e-6) & (coords[2] > 50.4e-6)
    avg_dT = np.mean(dT_fem[wg_mask])

    return coords, dT_fem, avg_dT


if __name__ == "__main__":
    coords_si, dT_si, avg_si = run_3d_fem_thermal(148.0)
    coords_bas, dT_bas, avg_bas = run_3d_fem_thermal(1300.0)

    print("=== 3D FEM SIMULATION RESULTS (Single Tile) ===")
    print(f"Si Substrate  (k = 148 W/m*K):  Local waveguide overheating = {avg_si:.3f} K")
    print(f"BAs Substrate (k = 1300 W/m*K): Local waveguide overheating = {avg_bas:.3f} K")
    print(f"Heat dissipation efficiency improvement: {avg_si / avg_bas:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    top_mask = np.isclose(coords_si[2], np.max(coords_si[2]))

    x_um = coords_si[0][top_mask] * 1e6
    y_um = coords_si[1][top_mask] * 1e6
    vmax = np.max(dT_si[top_mask])

    for ax, dT, title in zip(
        axes,
        [dT_si[top_mask], dT_bas[top_mask]],
        [f"Si Substrate (ΔT_wg = {avg_si:.2f} K)", f"BAs Substrate (ΔT_wg = {avg_bas:.2f} K)"]
    ):
        tc = ax.tricontourf(x_um, y_um, dT, levels=40, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("X, μm (Length from center)")
        ax.set_ylabel("Y, μm (Width from center)")
        ax.plot([0, 100, 100], [7.5, 7.5, 0], "c--", linewidth=1.5, label="MZI Heater Boundary")
        ax.legend(loc="upper right")

    cbar = fig.colorbar(tc, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Local Overheating ΔT (K)", fontsize=11)
    plt.suptitle("3D FEM Verification: Surface Thermal Profile of Photonic Chip", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fem_3d_profile.png", dpi=300)
    plt.show()

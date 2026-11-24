import numpy as np

def solve_3d_thermal_fdm(k_sub, rho_cp, pitch_um=25.0, t_sw=5e-6):
    nx, ny, nz = 26, 41, 27
    dx, dy, dz = 10e-6, 2.5e-6, 2e-6
    k = np.full((nx, ny, nz), k_sub, dtype=np.float64)
    k[:, :, -1] = 1.4
    Q = np.zeros((nx, ny, nz), dtype=np.float64)
    P_heater = 15e-3
    vol_heater = 100e-6 * 7.5e-6 * dz
    Q[:11, :4, -1] = (P_heater / 4.0) / vol_heater
    T = np.zeros((nx, ny, nz), dtype=np.float64)

    kx_p = 2 * k[:-1, :, :] * k[1:, :, :] / (k[:-1, :, :] + k[1:, :, :]) / (dx ** 2)
    ky_p = 2 * k[:, :-1, :] * k[:, 1:, :] / (k[:, :-1, :] + k[:, 1:, :]) / (dy ** 2)
    kz_p = 2 * k[:, :, :-1] * k[:, :, 1:] / (k[:, :, :-1] + k[:, :, 1:]) / (dz ** 2)

    for _ in range(1500):
        T_pad = np.pad(T, ((1, 1), (1, 1), (1, 1)), mode="edge")
        T_pad[:, :, 0] = 0.0
        flux_x = np.pad(kx_p, ((1, 0), (0, 0), (0, 0)), mode="edge") * (T_pad[:-2, 1:-1, 1:-1] - T) + \
                 np.pad(kx_p, ((0, 1), (0, 0), (0, 0)), mode="edge") * (T_pad[2:, 1:-1, 1:-1] - T)
        flux_y = np.pad(ky_p, ((0, 0), (1, 0), (0, 0)), mode="edge") * (T_pad[1:-1, :-2, 1:-1] - T) + \
                 np.pad(ky_p, ((0, 0), (0, 1), (0, 0)), mode="edge") * (T_pad[1:-1, 2:, 1:-1] - T)
        flux_z = np.pad(kz_p, ((0, 0), (0, 0), (1, 0)), mode="edge") * (T_pad[1:-1, 1:-1, :-2] - T) + \
                 np.pad(kz_p, ((0, 0), (0, 0), (0, 1)), mode="edge") * (T_pad[1:-1, 1:-1, 2:] - T)
        denom = np.pad(kx_p, ((1, 0), (0, 0), (0, 0)), mode="edge") + np.pad(kx_p, ((0, 1), (0, 0), (0, 0)), mode="edge") + \
                np.pad(ky_p, ((0, 0), (1, 0), (0, 0)), mode="edge") + np.pad(ky_p, ((0, 0), (0, 1), (0, 0)), mode="edge") + \
                np.pad(kz_p, ((0, 0), (0, 0), (1, 0)), mode="edge") + np.pad(kz_p, ((0, 0), (0, 0), (0, 1)), mode="edge")
        T = T + 0.8 * ((flux_x + flux_y + flux_z + Q) / denom)
        T[:, :, 0] = 0.0

    idx_nbr = int(round((pitch_um * 1e-6) / dy))
    dT_act = float(np.mean(T[:11, :4, -1]))
    dT_nbr = float(np.mean(T[:11, max(0, idx_nbr - 2):idx_nbr + 3, -1]))
    xtalk = dT_nbr / dT_act
    alpha = k_sub / rho_cp
    tau_diff = (50e-6 ** 2) / alpha
    mem = float(np.exp(-t_sw / (tau_diff + 5.5e-6)))
    return dT_act, dT_nbr, xtalk, mem

if __name__ == "__main__":
    print("Material | dT_act (K) | dT_nbr (K) | Crosstalk | Tile Memory")
    print("-" * 64)
    for name, k, rcp in [("Si", 148.0, 1.63e6), ("BAs", 1300.0, 1.85e6)]:
        da, dn, xt, mem = solve_3d_thermal_fdm(k, rcp)
        print(f"{name:8s} | {da:10.3f} | {dn:10.3f} | {xt:9.3f} | {mem:11.3f}")

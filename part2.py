import numpy as np
from part1 import VDW, TARGETS, SUBS, find_pdb, load_pdb_targeted, build_w

def build_x(poses, px, r0, a=0.88):
    M, K = len(poses), len(r0)
    r12, r6, rm = np.zeros((K, M)), np.zeros((K, M)), a * r0
    for c in range(M):
        d = np.linalg.norm(px[:, None, :] - poses[c][None, :, :], axis=2).reshape(-1)
        re = np.maximum(d, rm)
        r12[:, c], r6[:, c] = re ** (-12), re ** (-6)
    return np.vstack((r12, r6))

def quant(X, b=8):
    K, l, Xq = len(X) // 2, (2 ** b) - 1, np.zeros_like(X)
    for s in (slice(0, K), slice(K, 2 * K)):
        mx = np.max(np.abs(X[s]))
        Xq[s] = X[s] if mx == 0 else np.round((X[s] / mx) * l) / l * mx
    return Xq

def therm(ks, na=100, ph=0.015, t_sw=5e-6):
    ah = 200e-6 * 15e-6
    rl, ra = np.sqrt(ah / np.pi), np.sqrt(na * ah * 4 / np.pi)
    dt = 1.25 * ph * (0.3e-6 / (1.4 * ah) + 1e-8 / ah + 1 / (2 * np.pi * ks * rl)) + (na * ph) / (2 * np.pi * ks * ra)
    dp = (2 * np.pi / 1550e-9) * 1.86e-4 * dt * 200e-6
    alpha = ks / 1.75e6
    tau_diff = (50e-6 ** 2) / alpha
    mem_factor = float(np.exp(-t_sw / (tau_diff + 5.5e-6)))
    xtalk_sigma = 0.164 * (148.0 / ks) ** 0.85
    return dt, dp, mem_factor, float(xtalk_sigma)

def sim(W, Xq, dp, mc, xs, na=100, b=8, dn=0.025, lim=25.0):
    K = len(W) // 2
    _, M = Xq.shape
    l = (2 ** b) - 1
    ea = np.zeros(M)
    for s in (slice(0, K), slice(K, 2 * K)):
        wc, xc = W[s], Xq[s, :]
        mx = np.max(np.abs(wc))
        if mx == 0:
            continue
        wq = np.round((wc / mx) * l) / l
        pt, co, pl = np.arccos(np.clip(wq, -1, 1)), np.zeros(M), 0.0
        for st in range(0, K, na):
            en = min(st + na, K)
            dd = dp * (mc * pl + np.random.normal(0, xs, en - st))
            co += np.cos(pt[st:en] + dd) @ xc[st:en, :]
            pl = np.mean(np.abs(wq[st:en]))
        ea += co * mx
    ad = np.clip(ea, -lim, lim)
    return ad + np.random.normal(0, dn * (np.std(ad) or 1e-6), ad.shape)

def rot_mat(ax, ay, az):
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ rx @ ry

def gen_poses_targeted(lx, bc, M=2000, R=3.5):
    lc, gr, ps = lx - np.mean(lx, axis=0), (1 + np.sqrt(5)) / 2, []
    n_near = max(10, int(0.02 * M))
    for _ in range(n_near):
        ax, ay, az = np.random.normal(0, 0.18, 3)
        ps.append((lc @ rot_mat(ax, ay, az).T) + bc + np.random.normal(0, 0.45, 3))
    for i in range(M - n_near):
        t = 2 * np.pi * i / gr
        p = np.arccos(1 - 2 * (i + 0.5) / (M - n_near))
        s = 2 * np.pi * i * 7 / (M - n_near)
        rs = R * (((i * 0.61803398875) % 1) ** (1 / 3))
        sh = np.array([np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)]) * rs
        ps.append((lc @ rot_mat(p, s, t).T) + bc + sh)
    return ps

def calc_ef(scores, is_active, top_frac=0.05):
    M = len(scores)
    k = max(1, int(M * top_frac))
    top_idx = np.argsort(scores)[:k]
    total_act = np.sum(is_active)
    if total_act == 0:
        return 0.0
    return (np.sum(is_active[top_idx]) / k) / (total_act / M)

rmsd = lambda a, b: np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))

if __name__ == "__main__":
    np.random.seed(42)
    sp = {n: therm(k) for n, k in SUBS}
    print(f"{'PDB':5s} | {'Si EF5%':7s} | {'BAs EF5%':8s} | {'CPU EF5%':8s} | {'BAs Native-RMSD (A)':19s}")
    print("-" * 62)
    agg_si, agg_bas, agg_cpu, agg_rmsd = [], [], [], []
    for pid, lc in TARGETS:
        fp = find_pdb(pid)
        px, pe, lx, le, bc = load_pdb_targeted(fp, lc, npock=60)
        ps = gen_poses_targeted(lx, bc, M=2000, R=3.5)
        rmsds = np.array([rmsd(p, lx) for p in ps])
        is_act = rmsds < 2.0
        _, r0 = build_w(pe, le)
        X = build_x(ps, px, r0)
        Xq = quant(X)
        ef_si, ef_bas, ef_cpu, nrmsd = [], [], [], []
        for _ in range(100):
            W, _ = build_w(pe, le, np.random.uniform(0.85, 1.15), np.random.uniform(0.95, 1.05))
            ef_cpu.append(calc_ef(W @ X, is_act, 0.05))
            _, dp_s, mc_s, xs_s = sp["Si"]
            ef_si.append(calc_ef(sim(W, Xq, dp_s, mc_s, xs_s), is_act, 0.05))
            _, dp_b, mc_b, xs_b = sp["BAs"]
            e_bas = sim(W, Xq, dp_b, mc_b, xs_b)
            ef_bas.append(calc_ef(e_bas, is_act, 0.05))
            nrmsd.append(np.min(rmsds[np.argsort(e_bas)[:5]]))
        agg_si.append(np.mean(ef_si))
        agg_bas.append(np.mean(ef_bas))
        agg_cpu.append(np.mean(ef_cpu))
        agg_rmsd.append(np.mean(nrmsd))
        print(f"{pid:5s} | {agg_si[-1]:7.2f} | {agg_bas[-1]:8.2f} | {agg_cpu[-1]:8.2f} | {agg_rmsd[-1]:19.2f}")
    print("-" * 62)
    print(f"{'MEAN':5s} | {np.mean(agg_si):7.2f} | {np.mean(agg_bas):8.2f} | {np.mean(agg_cpu):8.2f} | {np.mean(agg_rmsd):19.2f}")

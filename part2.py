import numpy as np
from part1 import *


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


def therm(ks, na=100, ph=0.015):
    ah = 200e-6 * 15e-6
    rl, ra = np.sqrt(ah / np.pi), np.sqrt(na * ah * 4 / np.pi)
    dt = 1.25 * ph * (0.3e-6 / (1.4 * ah) + 1e-8 / ah + 1 / (2 * np.pi * ks * rl)) + (na * ph) / (2 * np.pi * ks * ra)
    dp = (2 * np.pi / 1550e-9) * 1.86e-4 * dt * 200e-6
    sc = np.sqrt(148.0 / ks)
    return dt, dp, 0.40 * sc, 0.20 * sc


def sim(W, Xq, dp, mc, xs, na=100, b=8, dn=0.025, lim=25.0):
    K, M, ea, l = len(W) // 2, Xq.shape[1], np.zeros(Xq.shape[1]), (2 ** b) - 1
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


def gen_poses(lx, bc, M=500, R=5.0):
    lc, gr, ps = lx - np.mean(lx, axis=0), (1 + np.sqrt(5)) / 2, []
    for i in range(M):
        t, p, s = 2 * np.pi * i / gr, np.arccos(1 - 2 * (i + 0.5) / M), 2 * np.pi * i * 7 / M
        rz = [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        rx = [[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]]
        ry = [[np.cos(s), 0, np.sin(s)], [0, 1, 0], [-np.sin(s), 0, np.cos(s)]]
        rs = R * (((i * 0.61803398875) % 1) ** (1 / 3))
        sh = np.array([np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)]) * rs
        ps.append((lc @ (np.array(rz) @ rx @ ry).T) + bc + sh)
    return ps


rmsd = lambda a, b: np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


if __name__ == "__main__":
    sp = {n: therm(k) for n, k in SUBS}
    for n, k in SUBS:
        print(f"{n:6s}: dT={sp[n][0]:.2f} K  dPhi={sp[n][1]:.3f} rad")
    print("-" * 65)
    agg = {n: [] for n, _ in SUBS}
    cr, nr = [], []
    for pid, lc in TARGETS:
        fp = find_pdb(pid)
        px, pe, lx, le, bc = load_pdb(fp, lc)
        ps = gen_poses(lx, bc)
        _, r0 = build_w(pe, le)
        X = build_x(ps, px, r0)
        Xq = quant(X)
        h = {n: 0 for n, _ in SUBS}
        c_l, n_l = [], []
        for _ in range(100):
            W, _ = build_w(pe, le, np.random.uniform(0.85, 1.15), np.random.uniform(0.95, 1.05))
            bt = np.argmin(W @ X)
            for n, _ in SUBS:
                t3 = np.argsort(sim(W, Xq, sp[n][1], sp[n][2], sp[n][3]))[:3]
                if bt in t3:
                    h[n] += 1
                if n == "BAs":
                    c_l.append(min(rmsd(ps[i], ps[bt]) for i in t3))
                    n_l.append(min(rmsd(ps[i], lx) for i in t3))
        for n in h:
            agg[n].append(h[n])
        cr.append(np.mean(c_l))
        nr.append(np.mean(n_l))
        print(f"{pid} | Si:{h['Si']:4.1f} AlN:{h['AlN']:4.1f} SiC:{h['4H-SiC']:4.1f} BAs:{h['BAs']:4.1f} | RMSD:{cr[-1]:.2f}/{nr[-1]:.2f}")
    print("-" * 65)
    print(f"MEAN | Si:{np.mean(agg['Si']):4.1f} AlN:{np.mean(agg['AlN']):4.1f} SiC:{np.mean(agg['4H-SiC']):4.1f} BAs:{np.mean(agg['BAs']):4.1f} | RMSD:{np.mean(cr):.2f}/{np.mean(nr):.2f}")

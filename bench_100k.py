import time
import json
import numpy as np
from part1 import find_pdb, load_pdb_targeted, build_w
from part2 import build_x, quant, therm, sim, rot_mat, rmsd

def gen_batch_targeted(lc, bc, batch_size, offset, total_M, R=3.5, n_near_total=100):
    gr = (1 + np.sqrt(5)) / 2
    ps = []
    for idx in range(offset, offset + batch_size):
        if idx < n_near_total:
            ax, ay, az = np.random.normal(0, 0.18, 3)
            ps.append((lc @ rot_mat(ax, ay, az).T) + bc + np.random.normal(0, 0.45, 3))
        else:
            i = idx - n_near_total
            m_bg = total_M - n_near_total
            t = 2 * np.pi * i / gr
            p = np.arccos(1 - 2 * (i + 0.5) / m_bg)
            s = 2 * np.pi * i * 7 / m_bg
            rs = R * (((i * 0.61803398875) % 1) ** (1 / 3))
            sh = np.array([np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)]) * rs
            ps.append((lc @ rot_mat(p, s, t).T) + bc + sh)
    return ps

if __name__ == "__main__":
    np.random.seed(42)
    pid, lig = "2VUK", "P83"
    TOTAL_M, BATCH = 100000, 2500
    fp = find_pdb(pid)
    px, pe, lx, le, bc = load_pdb_targeted(fp, lig, npock=60)
    lc = lx - np.mean(lx, axis=0)
    W, r0 = build_w(pe, le)
    _, dp_si, mc_si, xs_si = therm(148.0)
    _, dp_bas, mc_bas, xs_bas = therm(1300.0)

    e_si_all, e_bas_all, nrmsd_all = [], [], []
    cpu_time_total = 0.0

    for off in range(0, TOTAL_M, BATCH):
        ps = gen_batch_targeted(lc, bc, BATCH, off, TOTAL_M)
        X = build_x(ps, px, r0)
        Xq = quant(X)
        t0 = time.perf_counter()
        _ = W @ X
        cpu_time_total += (time.perf_counter() - t0) * 1000.0
        e_si_all.append(sim(W, Xq, dp_si, mc_si, xs_si))
        e_bas_all.append(sim(W, Xq, dp_bas, mc_bas, xs_bas))
        nrmsd_all.extend([rmsd(p, lx) for p in ps])

    e_si_all = np.concatenate(e_si_all)
    e_bas_all = np.concatenate(e_bas_all)
    nrmsd_all = np.array(nrmsd_all)

    n_tiles = int(np.ceil(len(W) / 100.0))
    bas_time_ms = n_tiles * (5e-6 + TOTAL_M / 10e9) * 1000.0
    speedup = cpu_time_total / bas_time_ms

    top5_bas = np.argsort(e_bas_all)[:5]
    best_nrmsd_bas = float(np.min(nrmsd_all[top5_bas]))

    is_active = nrmsd_all < 2.0
    top1pct = int(TOTAL_M * 0.01)
    ef1_bas = float((np.sum(is_active[np.argsort(e_bas_all)[:top1pct]]) / top1pct) / (np.sum(is_active) / TOTAL_M))
    ef1_si = float((np.sum(is_active[np.argsort(e_si_all)[:top1pct]]) / top1pct) / (np.sum(is_active) / TOTAL_M))

    res = {
        "target": pid,
        "total_M": TOTAL_M,
        "cpu_ms": round(cpu_time_total, 2),
        "bas_ms": round(bas_time_ms, 3),
        "speedup": round(speedup, 1),
        "native_rmsd_bas_A": round(best_nrmsd_bas, 2),
        "ef1_bas": round(ef1_bas, 2),
        "ef1_si": round(ef1_si, 2),
    }
    with open("data_100k.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

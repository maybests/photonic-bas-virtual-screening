import numpy as np
from part1 import find_pdb, load_pdb_targeted, build_w
from part2 import build_x, quant, therm, sim, gen_poses_targeted, calc_ef, rmsd

if __name__ == "__main__":
    np.random.seed(42)
    fp = find_pdb("1M17")
    px, pe, lx, le, bc = load_pdb_targeted(fp, "AQ4", npock=60)
    ps = gen_poses_targeted(lx, bc, M=2000, R=3.5)
    rmsds = np.array([rmsd(p, lx) for p in ps])
    is_act = rmsds < 2.0

    _, r0 = build_w(pe, le)
    X = build_x(ps, px, r0)
    Xq = quant(X)

    print("K_sub | dT (K) | dPhi  | EF5% (x)")
    print("-" * 38)
    for k in range(100, 1401, 150):
        dt, dp, mc, xs = therm(k)
        efs = []
        for _ in range(100):
            W, _ = build_w(pe, le, np.random.uniform(0.85, 1.15), np.random.uniform(0.95, 1.05))
            e_opt = sim(W, Xq, dp, mc, xs)
            efs.append(calc_ef(e_opt, is_act, 0.05))
        print(f"{k:5d} | {dt:6.2f} | {dp:5.3f} | {np.mean(efs):7.2f}x")

import numpy as np
from part1 import *
from part2 import build_x, quant, therm, sim, gen_poses

if __name__ == "__main__":
    fp = find_pdb("1M17")
    px, pe, lx, le, bc = load_pdb(fp, "AQ4", 40)
    ps = gen_poses(lx, bc, M=500, R=5.0)
    _, r0 = build_w(pe, le)
    X = build_x(ps, px, r0)
    Xq = quant(X)

    print("K_sub | dT (K) | dPhi (rad) | Top3_Acc (%)")
    print("-" * 44)
    for k in range(100, 1401, 150):
        dt, dp, mc, xs = therm(k)
        hits = 0
        for _ in range(100):
            W, _ = build_w(pe, le, np.random.uniform(0.85, 1.15), np.random.uniform(0.95, 1.05))
            bt = np.argmin(W @ X)
            if bt in np.argsort(sim(W, Xq, dp, mc, xs))[:3]:
                hits += 1
        print(f"{k:5d} | {dt:6.2f} | {dp:10.3f} | {hits:10.1f}%")

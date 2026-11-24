import os
import urllib.request
import numpy as np

np.random.seed(42)

VDW = {
    "C": (1.938, 0.086),
    "N": (1.824, 0.170),
    "O": (1.661, 0.210),
    "S": (2.000, 0.250),
}

TARGETS = [
    ("2VUK", "P83"), ("1HSG", "MK1"),
    ("3ERT", "OHT"), ("1M17", "AQ4"),
    ("1IEP", "STI"), ("2HYY", "STI"),
    ("1EVE", "E20"), ("1CX2", "S58"),
    ("1FJS", "Z34"), ("1U1C", "BAU"),
]

SUBS = [
    ("Si", 148.0),
    ("AlN", 285.0),
    ("4H-SiC", 370.0),
    ("BAs", 1300.0),
]


def find_pdb(pid):
    dirs = [".", "pdb"]
    for d in dirs:
        for n in (pid + ".pdb", pid.lower() + ".pdb"):
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    os.makedirs("pdb", exist_ok=True)
    url = f"https://rcsb.org{pid}.pdb"
    dst = os.path.join("pdb", f"{pid}.pdb")
    urllib.request.urlretrieve(url, dst)
    return dst


def find_blind_pocket(px, step=2.0):
    x0, y0, z0 = np.min(px, axis=0) + 4.0
    x1, y1, z1 = np.max(px, axis=0) - 4.0
    g = np.mgrid[x0:x1:step, y0:y1:step, z0:z1:step].reshape(3, -1).T
    if len(g) == 0:
        return np.mean(px, axis=0)
    best_pt, best_sc = np.mean(px, axis=0), -1
    for pt in g:
        d = np.linalg.norm(px - pt, axis=1)
        if np.min(d) < 3.2:
            continue
        sc = np.sum((d >= 4.0) & (d <= 9.0))
        if sc > best_sc:
            best_sc, best_pt = sc, pt
    return best_pt


def load_pdb(fp, lc, npock=40):
    pc, pe, lco, le, lch = [], [], [], [], None
    with open(fp, "r", errors="ignore") as f:
        for l in f:
            r = l[:6].strip()
            if r not in ("ATOM", "HETATM") or l[16:17] not in (" ", "A", ""):
                continue
            el = l[76:78].strip().upper() or "".join(c for c in l[12:16] if c.isalpha())[:1]
            if el not in VDW:
                continue
            xyz = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
            if r == "HETATM" and l[17:20].strip() == lc:
                if lch is None:
                    lch = l[21:22]
                if l[21:22] == lch:
                    lco.append(xyz)
                    le.append(el)
            elif r == "ATOM":
                pc.append(xyz)
                pe.append(el)
    px, lx = np.array(pc), np.array(lco)
    bc = find_blind_pocket(px)
    idx = np.argsort(np.linalg.norm(px - bc, axis=1))[:npock]
    return px[idx], [pe[i] for i in idx], lx, le, bc


def build_w(pe, le, bs=1.0, rs=1.0):
    al, bl, r0l = [], [], []
    for ep in pe:
        rp, vp = VDW[ep]
        for el in le:
            rl, vl = VDW[el]
            r0 = (rp + rl) * rs
            eps = np.sqrt(vp * vl) * bs
            al.append(eps * (r0 ** 12))
            bl.append(2.0 * eps * (r0 ** 6))
            r0l.append(r0)
    return np.concatenate((al, -np.array(bl))), np.array(r0l)

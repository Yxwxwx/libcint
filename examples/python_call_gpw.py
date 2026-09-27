"""Call all five GPW integral types from the libcint source checkout.

Requires PySCF, NumPy, and a built libgpw shared library. Set GPW_LIBRARY
to its absolute path if it is not installed or built in libcint/build.
"""

import sys
from pathlib import Path

import numpy as np
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import cint_gpw as gpw


mol = gto.M(
    atom="O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266",
    unit="Bohr",
    basis="sto-3g",
    verbose=0,
)
k = np.array([[0.1, 0.2, -0.1], [0.3, -0.2, 0.5]])  # bohr^-1
kp = k[::-1].copy()  # [k[1], k[0]]
nao = mol.nao_nr()

with gpw.prepare(mol) as opt:
    overlap = gpw.ovlp(mol, k, opt=opt)    # <G_a|P_k>: (nk, nao)
    kinetic = gpw.kin(mol, k, opt=opt)    # <G_a|-∇²/2|P_k>: (nk, nao)
    nuclear = gpw.nuc(mol, k, opt=opt)    # <G_a|V_nuc|P_k>: (nk, nao)
    gpgg = gpw.gpgg(mol, k, opt=opt)      # (G_a P_k|G_b G_c): (nk, nao, nao, nao)
    # k[i] pairs with kp[i]; the second plane wave is conjugated internally.
    gppg = gpw.gppg(mol, k, kp, opt=opt) # (G_a P_k|P_kp G_b): (nk, nao, nao)
    gppg_pair = gpw.gppg(mol, k[0], k[1], opt=opt)  # (G_a P_k[0]|P_k[1] G_b): (nao, nao)
    # All four (k[i], k[j]) combinations, indexed as [i, j, AO_a, AO_b].
    k_pairs = np.repeat(k, len(k), axis=0)
    kp_pairs = np.tile(k, (len(k), 1))
    gppg_all = gpw.gppg(mol, k_pairs, kp_pairs, opt=opt).reshape(len(k), len(k), nao, nao)

assert overlap.shape == kinetic.shape == nuclear.shape == (len(k), nao)
assert gpgg.shape == (len(k), nao, nao, nao)
assert gppg.shape == (len(k), nao, nao)
assert gppg_pair.shape == (nao, nao)
assert gppg_all.shape == (len(k), len(k), nao, nao)
np.testing.assert_allclose(gppg_pair, gppg[0])
np.testing.assert_allclose(gppg_pair, gppg_all[0, 1])
np.testing.assert_allclose(kinetic, 0.5 * np.sum(k * k, axis=1)[:, None] * overlap)

for name, value in (
    ("ovlp", overlap), ("kin", kinetic), ("nuc", nuclear),
    ("gpgg", gpgg), ("gppg", gppg),
    ("gppg(k[0], k[1])", gppg_pair), ("gppg(all pairs)", gppg_all),
):
    print(f"{name}: shape={value.shape}, first={value.flat[0]:.12g}")

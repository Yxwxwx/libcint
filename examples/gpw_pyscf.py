"""Run after building WITH_GPW=ON and installing ./python (see GPW_README.md)."""
import numpy as np
from pyscf import gto
import cint_gpw as gpw

mol = gto.M(atom='H 0 0 -.7; H 0 0 .7', unit='Bohr', basis='cc-pvdz', verbose=0)
k = np.array([[0., 0., 0.], [.3, -.2, .5]])  # bohr^-1
kp = np.array([[.1, .2, -.1], [-.4, .3, .2]])
with gpw.prepare(mol) as opt:
    s = gpw.ovlp(mol, k, opt=opt)
    t = gpw.kin(mol, k, opt=opt)
    v = gpw.nuc(mol, k, opt=opt)

    # Chemists' (G_a P_k | G_b G_c), all AO indices, batched over k.
    gpgg = gpw.gpgg(mol, k, opt=opt)          # (nk, nao, nao, nao)
    # Chemists' (G_a P_k | P_kp G_b). k[i] pairs with kp[i]; the
    # second PW is conjugated internally, so pass its actual wavevector kp.
    gppg = gpw.gppg(mol, k, kp, opt=opt)     # (nk, nao, nao)

    # Shell blocks at a single wavevector/pair (no leading batch axis).
    gpgg_shell = gpw.gpgg_by_shell(mol, (0, 1, 2), k[1], opt=opt)
    gppg_shell = gpw.gppg_by_shell(mol, (0, 1), k[1], kp[1], opt=opt)
    loc = mol.ao_loc_nr()
    a, b, c = (slice(loc[i], loc[i+1]) for i in (0, 1, 2))
    assert np.allclose(gpgg_shell, gpgg[1, a, b, c])
    assert np.allclose(gppg_shell, gppg[1, a, b])
    assert np.allclose(t, .5*(k*k).sum(axis=1)[:, None]*s)
    assert np.array_equal(s[:, :1], gpw.ovlp_by_shell(mol, (0,), k, opt=opt))
    print('S/T/V:', s.shape, t.shape, v.shape)
    print('(G P|G G):', gpgg.shape, '(G P|P G):', gppg.shape)
    print('Single-wavevector shell blocks:', gpgg_shell.shape, gppg_shell.shape)
    print('(G_0 P_k[1]|G_1 G_2):', gpgg_shell[0, 0, 0])
    print('(G_0 P_k[1]|P_kp[1] G_1):', gppg_shell[0, 0])
    print('max |S|, |T|, |V|:', *(float(np.max(abs(a))) for a in (s, t, v)))

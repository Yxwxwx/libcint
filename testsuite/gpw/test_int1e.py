"""Independent GPW checks. Run with Python + numpy, mpmath and PySCF.

The reference uses Fourier derivatives and explicit polynomial Boys moments,
not the C recurrence or its quadrature nodes. PySCF's original GG integrals
provide a second, exact reference at k=0 by splitting each exponent in half.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import json
import math
import os
from pathlib import Path
import sys
import time

import mpmath as mp
import numpy as np
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'python'))
import cint_gpw as gpw


def powers(l):
    return [(x, y, l-x-y) for x in range(l, -1, -1) for y in range(l-x, -1, -1)]


def product(a, b):
    c = [mp.mpc(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i+j] += x*y
    return c


def linear_power(a, b, n):
    return [math.comb(n, j)*a**(n-j)*b**j for j in range(n+1)]


def moment_polynomial(l, shift, distance, p, tau2):
    # Explicit Gaussian moments, as a polynomial in s=u^2, with no recurrence.
    result = [mp.mpc(0)]*(l+1)
    for j in range(l//2+1):
        coeff = mp.mpf(math.factorial(l))/(math.factorial(l-2*j)*math.factorial(j)*2**j)
        terms = product(linear_power(shift, -tau2*distance, l-2*j),
                        linear_power(1/(2*p), -tau2/(2*p), j))
        for h, v in enumerate(terms):
            result[h] += coeff*v
    return result


def reference(mol, shell, k, op, dps=60):
    with mp.workdps(dps):
        b = mol._bas[shell]
        l, np_, nc = map(int, b[[1, 2, 3]])
        center = list(map(mp.mpf, mol.atom_coord(int(b[0]))))
        wave = list(map(mp.mpf, k))
        phase = mp.exp(1j*sum(x*y for x, y in zip(center, wave)))
        sp = 1/mp.sqrt(4*mp.pi) if l == 0 else mp.sqrt(3/(4*mp.pi)) if l == 1 else 1
        out = [[mp.mpc(0) for _ in powers(l)] for _ in range(nc)]
        for ip in range(np_):
            p = mp.mpf(mol._env[b[5]+ip])
            x = sum(v*v for v in wave)/(4*p)
            shift = [1j*v/(2*p) for v in wave]
            val = [mp.mpc(0) for _ in powers(l)]
            if op != 'nuc':
                fourier = [[(-1j)**n*mp.diff(lambda q: mp.sqrt(mp.pi/p)*mp.exp(-q*q/(4*p)), wave[d], n)
                            for n in range(l+1)] for d in range(3)]
                for j, xyz in enumerate(powers(l)):
                    val[j] = phase*sp*mp.fprod(fourier[d][n] for d, n in enumerate(xyz))
                if op == 'kin':
                    val = [v*sum(q*q for q in wave)/2 for v in val]
            else:
                for atom in mol._atm:
                    model = int(atom[2])
                    charge = mp.mpf(mol._env[atom[4]]) if model == 3 else abs(int(atom[0]))
                    if not charge:
                        continue
                    zeta = mp.mpf(mol._env[atom[3]]) if model == 2 else mp.mpf(0)
                    tau2 = zeta/(p+zeta) if zeta > 0 else mp.mpf(1)
                    distance = [center[d]-mp.mpf(mol._env[atom[1]+d])+shift[d] for d in range(3)]
                    t = p*tau2*sum(d*d for d in distance)
                    boys = [mp.hyp1f1(mp.mpf(n)+mp.mpf('.5'), mp.mpf(n)+mp.mpf('1.5'), -t)/(2*n+1)
                            for n in range(l+1)]
                    poly = [[moment_polynomial(n, shift[d], distance[d], p, tau2)
                             for n in range(l+1)] for d in range(3)]
                    pref = -charge*2*mp.pi/p*mp.sqrt(tau2)*sp*phase*mp.exp(-x)
                    for j, xyz in enumerate(powers(l)):
                        terms = product(product(poly[0][xyz[0]], poly[1][xyz[1]]), poly[2][xyz[2]])
                        val[j] += pref*sum(v*boys[n] for n, v in enumerate(terms))
            for c in range(nc):
                coeff = mp.mpf(mol._env[b[6]+c*np_+ip])
                for j, v in enumerate(val):
                    out[c][j] += coeff*v
        return np.array(out, dtype=complex).ravel()


def pyscf_zero(mol, shell, op):
    b = mol._bas[shell]
    l, np_, nc = map(int, b[[1, 2, 3]])
    result = np.zeros((nc, (l+1)*(l+2)//2))
    for ip in range(np_):
        offset = len(mol._env)
        env = np.r_[mol._env, mol._env[b[5]+ip]/2, 1., math.sqrt(4*math.pi)]
        bas = np.array([[b[0], l, 1, 1, 0, offset, offset+1, 0],
                        [b[0], 0, 1, 1, 0, offset, offset+2, 0]], dtype=np.int32)
        value = gto.moleintor.getints_by_shell('int1e_'+op+'_cart', (0, 1), mol._atm, bas, env)[:, 0]
        for c in range(nc):
            result[c] += mol._env[b[6]+c*np_+ip]*value
    return result.ravel()


def molecule(l, model=1):
    m = gto.M(atom='He .2 -.3 .1; He .7 .1 -.4', unit='Bohr', verbose=0,
              basis={'He': [[l, [1.3, .6, .2], [.4, .3, -.1]]]})
    if model != 1:
        start = len(m._env)
        m._env = np.r_[m._env, .8, -0.4]
        m._atm[:, 2] = model
        m._atm[:, 3 if model == 2 else 4] = [start, start+1]
        if model == 2:
            m._env[-1] = 2.1
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library')
    parser.add_argument('--report')
    args = parser.parse_args()
    if args.library:
        os.environ['GPW_LIBRARY'] = str(Path(args.library).resolve())
    started = time.perf_counter()
    summary = {'checks': 0, 'max_abs_error': 0., 'max_tolerance_ratio': 0.}

    def close(got, expected, label):
        got, expected = np.asarray(got), np.asarray(expected)
        assert got.shape == expected.shape, (label, got.shape, expected.shape)
        assert np.isfinite(got).all(), label
        error = abs(got-expected)
        ratio = error/(1e-11+1e-10*abs(expected))
        summary['checks'] += 1
        if error.size:
            summary['max_abs_error'] = max(summary['max_abs_error'], float(error.max()))
            summary['max_tolerance_ratio'] = max(summary['max_tolerance_ratio'], float(ratio.max()))
            assert ratio.max() <= 1, (label, error.max(), ratio.max())

    waves = np.array([[0., 0., 0.], [.3, -.2, .5], [4., 1., -2.]])
    for model in (1, 2, 3):
        for l in range(7):
            m = molecule(l, model)
            before = [a.copy() for a in (m._atm, m._bas, m._env)]
            with gpw.prepare(m) as opt:
                for op in ('ovlp', 'kin', 'nuc'):
                    f = getattr(gpw, op)
                    got = f(m, waves, shls=(0,), cart=True, opt=opt)
                    ref = np.array([reference(m, 0, k, op) for k in waves])
                    close(got, ref, f'mpmath {op} model={model} l={l}')
                    close(f(m, waves, shls=(0,), cart=True), got, 'opt')
                    close(f(m, waves, shls=(0,), cart=False, opt=opt),
                          (ref.reshape(3, 2, -1) @ gto.cart2sph(l, normalized='sp')).reshape(3, -1), 'spherical')
                    close(got, np.array([f(m, k, shls=(0,), cart=True, opt=opt) for k in waves]), 'batch')
                    close(f(m, -waves, shls=(0,), cart=True, opt=opt), got.conj(), 'conjugation')
                    if op != 'kin':
                        close(got[0], pyscf_zero(m, 0, op), f'PySCF {op} model={model} l={l}')
                    moved = m.copy()
                    moved._env = m._env.copy()
                    delta = np.array([.31, -.27, .19])
                    for ptr in set(moved._atm[:, 1]):
                        moved._env[ptr:ptr+3] += delta
                    close(f(moved, waves, shls=(0,), cart=True), got*np.exp(1j*waves@delta)[:, None], 'translation')
            for a, b in zip(before, (m._atm, m._bas, m._env)):
                assert np.array_equal(a, b), 'input mutated'
    print('All l=0..6, three nuclear models, contractions, PySCF and mpmath passed', flush=True)

    # Negative T through -10000: overlap underflows but point-nucleus tails do not.
    m = gto.M(atom='He 0 0 0', basis={'He': [[0, [1., 1.]]]}, verbose=0)
    for x in (0., 1e-9, 1., 100., 700., 1000., 10000.):
        k = np.array([2*math.sqrt(x), 0., 0.])
        close(gpw.nuc(m, k, cart=True), reference(m, 0, k, 'nuc', 80), f'scaled T={-x}')
    assert gpw.ovlp(m, [200., 0., 0.])[0] == 0
    assert abs(gpw.nuc(m, [200., 0., 0.])[0]) > 1e-4

    # Diffuse/tight primitives and higher-l negative-T tails, beyond the main set.
    for l in (0, 3, 6):
        for exponent in (.001, 1., 1000.):
            m = gto.M(atom='He .2 -.3 .1', unit='Bohr', verbose=0,
                      basis={'He': [[l, [exponent, 1.]]]})
            for factor in (.01, 3., 20., 200.):
                k = math.sqrt(exponent)*np.array([factor, -.2, .1])
                for op in ('ovlp', 'kin', 'nuc'):
                    close(getattr(gpw, op)(m, k, cart=True), reference(m, 0, k, op, 80),
                          f'extreme exponent={exponent} l={l} factor={factor} {op}')
    # Match upstream abs(integer charge), zero-charge ghosts and zeta=0 limit.
    m = molecule(2)
    m._atm[0, 0] = -2
    m._atm[1, 0] = 0
    close(gpw.nuc(m, waves[0], shls=(0,), cart=True), pyscf_zero(m, 0, 'nuc'), 'signed charge and ghost')
    m._atm[:, 2] = 2
    m._atm[:, 3] = 0  # env[0] = 0: Gaussian model reduces to a point nucleus.
    close(gpw.nuc(m, waves[0], shls=(0,), cart=True), pyscf_zero(m, 0, 'nuc'), 'zero zeta')

    # Direct Laplace integration is independent even of the Boys reference.
    m = molecule(4)
    k = waves[1]
    with mp.workdps(65):
        b = m._bas[0]
        total = mp.mpc(0)
        center = list(map(mp.mpf, m.atom_coord(0)))
        for ip in range(int(b[2])):
            p = mp.mpf(m._env[b[5]+ip])
            shift = [1j*mp.mpf(v)/(2*p) for v in k]
            phase = 1j*sum(mp.mpf(k[d])*center[d] for d in range(3))-sum(mp.mpf(v)**2 for v in k)/(4*p)
            for atom in m._atm:
                d = [center[h]+shift[h]-mp.mpf(m._env[atom[1]+h]) for h in range(3)]
                t = p*sum(v*v for v in d)
                # xxxx moment = mean^4 + 6 mean^2 variance + 3 variance^2.
                def integrand(u):
                    mean, var = shift[0]-u*u*d[0], (1-u*u)/(2*p)
                    return mp.exp(phase-t*u*u)*(mean**4+6*mean*mean*var+3*var*var)
                total += -int(atom[0])*2*mp.pi/p*mp.mpf(m._env[b[6]+ip])*mp.quad(integrand, [0, .5, 1])
        close(gpw.nuc(m, k, shls=(0,), cart=True)[:1], np.array([complex(total)]), 'direct Laplace integral')

    # Full AO placement, norms, empty/strided batches, and shared read-only opt.
    m = gto.M(atom='O 0 0 0; H 0 1.4 1.1; H 0 -1.4 1.1', unit='Bohr', basis='cc-pvdz', verbose=0)
    with gpw.prepare(m) as opt:
        for cart in (True, False):
            for op in ('ovlp', 'kin', 'nuc'):
                f = getattr(gpw, op)
                result = f(m, waves, cart=cart, opt=opt)
                loc = m.ao_loc_nr(cart=cart)
                for shell in range(m.nbas):
                    close(result[:, loc[shell]:loc[shell+1]], f(m, waves, shls=(shell,), cart=cart, opt=opt), 'AO placement')
                close(f(m, waves[:, ::-1], cart=cart, opt=opt), f(m, waves[:, ::-1].copy(), cart=cart, opt=opt), 'strides')
                close(f(m, waves, cart=cart, pw_norm='box', volume=8., opt=opt), result/math.sqrt(8), 'box')
                close(f(m, waves, cart=cart, pw_norm='delta3', opt=opt), result*(2*math.pi)**-1.5, 'delta3')
                assert f(m, np.empty((0, 3)), cart=cart, opt=opt).shape == (0, int(loc[-1]))
                assert result.dtype == np.complex128 and result.flags.c_contiguous
        close(gpw.kin(m, waves, opt=opt), gpw.ovlp(m, waves, opt=opt)*(waves*waves).sum(axis=1)[:, None]/2, 'kinetic')
        with ThreadPoolExecutor(4) as pool:
            for result in pool.map(lambda _: gpw.nuc(m, waves, opt=opt), range(12)):
                close(result, gpw.nuc(m, waves, opt=opt), 'shared opt threads')
        m._env[-1] += .01
        try:
            gpw.ovlp(m, waves, opt=opt)
        except RuntimeError as exc:
            assert 'changed' in str(exc)
        else:
            raise AssertionError('stale opt accepted')
        m._env[-1] -= .01

    # Trust boundary checks through C: validation cannot touch output.
    m = molecule(0)
    lib = gpw.library()
    shls = np.array([0], dtype=np.int32)
    output = np.full(8, 123+456j)
    def call(op=3, cart=1, capacity=6, wave=waves, n=3, env=None):
        arrays, args = gpw.inputs(m._atm, m._bas, m._env if env is None else env)
        return lib.gpw_int1e_batch(gpw.double_pointer(output), capacity, op, cart,
            gpw.int_pointer(shls), gpw.double_pointer(wave), n, *args, None, None)
    for options in ({'op': 0}, {'cart': 2}, {'capacity': 5}, {'env': m._env[:20]},
                    {'wave': np.full((3, 3), np.nan)}):
        assert call(**options) < 0
        assert np.all(output == 123+456j)
    # Second nucleus has no selected shell; its coordinates still require bounds checking.
    old = int(m._atm[1, 1])
    m._atm[1, 1] = len(m._env)
    assert call() == -1 and np.all(output == 123+456j)
    m._atm[1, 1] = old
    for model, ptr, val in ((2, len(m._env), 1), (3, len(m._env), 1), (2, 0, -1), (3, 0, float('nan'))):
        atom, env = m._atm.copy(), m._env.copy()
        m._atm[1, 2] = model
        m._atm[1, 3 if model == 2 else 4] = ptr
        m._env[0] = val
        assert call() == -1 and np.all(output == 123+456j)
        m._atm, m._env = atom, env
    for op in (1, 2, 3):
        output[:] = 123+456j
        bad = waves.copy()
        bad[-1, 0] = 1e308
        assert call(op=op, wave=bad) < 0
        assert np.all(output[:6] == 0) and np.all(output[6:] == 123+456j)
    for k in ([np.nan, 0, 0], [1j, 0, 0], [[1, 2]], [1, 2]):
        try:
            gpw.ovlp(m, k)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid wave accepted')
    # Standalone two-electron wrapper uses the same C ABI and AO layout.
    m = gto.M(atom='He 0 0 0', basis={'He': [[0, [1., 1.]], [1, [.7, 1.]]]}, verbose=0)
    with gpw.prepare(m) as opt:
        for kind in ('gpgg', 'gppg'):
            extra = (waves*.7,) if kind == 'gppg' else ()
            f = getattr(gpw, kind)
            got = f(m, waves, *extra, opt=opt)
            for j, k in enumerate(waves):
                close(got[j], f(m, k, *([k*.7] if kind == 'gppg' else []), opt=opt), '2e standalone batch')
            norm = (2*math.pi)**(-3 if kind == 'gppg' else -1.5)
            close(f(m, waves, *extra, opt=opt, pw_norm='delta3'), got*norm, '2e normalization')
    print('Scaled tails, Laplace quadrature, full AO, concurrency and boundary checks passed', flush=True)
    summary['seconds'] = time.perf_counter()-started
    summary['library'] = lib.gpw_build_version().decode()
    print(json.dumps(summary, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()

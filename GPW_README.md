# Standalone mixed GTO–PW integrals

This libcint fork provides `libgpw` and `include/gpw.h`. It computes GTO–PW
overlap, kinetic and nuclear attraction, and the mixed two-electron `gpgg` and
`gppg` integrals. The mixed kernels are in `src/gpw`, with libcint's original
real Rys and Cartesian-to-spherical sources compiled into `libgpw`. No
Mixed-GTO-PW checkout, C++ extension, BLAS or LAPACK is needed. As with ordinary
libcint, libquadmath is used when detected at build time. The ordinary `libcint` library retains
its original integral code and ABI. `WITH_GPW` defaults to `OFF`.

## Build and use with PySCF

From this repository's root:

```sh
cmake -S . -B build -DWITH_GPW=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python -m pip install ./python
export GPW_LIBRARY="$PWD/build/libgpw.so"
python examples/gpw_pyscf.py
```

The Python interpreter needs NumPy and PySCF. `cint_gpw` is a small ctypes
module; it uses PySCF's existing `mol._atm`, `mol._bas`, and `mol._env`. PySCF
itself needs no patch. Alternatively, set `PYTHONPATH="$PWD/python"` without
installing the module. `GPW_LIBRARY` must be absolute; when omitted, the module
tries the source tree's `build/libgpw.so`, then the system library search path.
`cmake --install build --prefix /your/prefix` installs the native library and
header; install the Python module separately as above.

```python
import numpy as np
from pyscf import gto
import cint_gpw as gpw

mol = gto.M(atom='H 0 0 -.7; H 0 0 .7', unit='Bohr', basis='cc-pvdz')
k = np.array([[0., 0., 0.], [.3, -.2, .5]])
with gpw.prepare(mol) as opt:
    S = gpw.ovlp(mol, k, opt=opt)       # (nk, nao)
    T = gpw.kin(mol, k, opt=opt)        # (nk, nao)
    V = gpw.nuc(mol, k, opt=opt)        # (nk, nao)
    H = T + V
    eri3 = gpw.gpgg(mol, k, opt=opt)    # (nk, nao, nao, nao)
    eri2 = gpw.gppg(mol, k, k, opt=opt) # (nk, nao, nao), zip pairs
    shell = gpw.nuc_by_shell(mol, (0,), k, opt=opt)
```

All five functions accept `cart=None` (use `mol.cart`), `cart=True` or
`cart=False`; `pw_norm='bare'`, `'delta3'`, `'box'` (requires `volume` in bohr³),
or `'legacy_k'`; and optional `opt`. A single `(3,)` wavevector removes the
outer batch axis. Empty `(0,3)` batches are valid. Outputs are C-contiguous
`complex128`. `*_by_shell(mol, shls, k, ...)`, or `shls=...` on the full function,
computes only the requested shell block. General contractions are retained.

`prepare` creates a GPW-specific immutable cache, **not** libcint's `CINTOpt`.
It stores the molecule and parsed shells/transforms, and rejects changed inputs
on every call. Multiple threads may share it; close it after all calls finish.
The C API also exposes reusable workspaces, one per concurrent caller. Each
Python call reuses one workspace across all its shell blocks; concurrent calls
have separate workspaces. Wavevector-independent primitive data is prepared
once per shell batch. General contractions accumulate one shell at a time;
single contractions share buffers and combine their coefficients with the
primitive prefactor. Workspaces hold three Cartesian buffers and a reusable
primitive-parameter table; `gpw_workspace_bytes` includes both.

## Definitions and normalization

Coordinates are bohr and wavevectors are bohr⁻¹. The core uses the bare plane
wave `P_k(r)=exp(+i k·r)`. Real GTOs follow libcint/PySCF normalization and AO
ordering, including the common s/p factors and all contraction columns.

| Entry | Integral |
| --- | --- |
| `ovlp` | `S_ak = <G_a | P_k>` |
| `kin` | `T_ak = <G_a | -∇²/2 | P_k> = (k²/2) S_ak` |
| `nuc` | `V_ak = <G_a | -Σ_A Z_A/r_A | P_k>` for point nuclei |
| `gpgg` | Chemists' `(G_a P_k | G_b G_c)` |
| `gppg` | Chemists' `(G_a P_k | P_kp G_b)`; second PW is conjugated internally |

Chemists' notation means `(ab|cd)=∫a*(r1)b(r1) r12⁻¹ c*(r2)d(r2) dr1 dr2`.
Zero/equal wavevectors are valid. The reversed one-electron block is the
conjugate transpose for these Hermitian operators. No electron spin factor,
exchange sign or two-electron factor of 1/2 is included.

Python multiplies each ket PW by its normalization and each bra PW by the
conjugate normalization: bare `1`, delta3 `(2π)^(-3/2)`, box `volume^(-1/2)`,
legacy_k `(2π)^(-3/2)|k|`. Choosing box normalization does not introduce periodic
Coulomb sums. The operators are free-space operators.

## Native API and supported domain

The six new single-wavevector symbols are
`gpw_int1e_{ovlp,kin,nuc}_gp_{cart,sph}`. `gpw_int1e_batch` selects `GPW_OVLP`,
`GPW_KIN` or `GPW_NUC`. All take one real GTO shell index; there is no dummy PW
shell. Existing `gpw_int2e_*` signatures are unchanged. See `include/gpw.h` and
the executable C example/check `testsuite/gpw/smoke.c` for complete declarations.

Indices are `int32_t`; outputs are interleaved real/imaginary doubles. Capacity
is measured in complex numbers. Within an AO axis, contraction precedes the
Cartesian/spherical component. Batches are outermost. A workspace needs at
least the Cartesian shell element count, including all contractions, even for
spherical output. `opt=NULL` and `work=NULL` are valid.

Supported: real GTOs with `l=0…6`, positive finite primitive exponents, general
contractions, Cartesian/PySCF real spherical AOs, and finite real wavevectors.
The nuclear operator follows `src/g1e.c`: point nuclei; Gaussian nuclei with
potential `-Z erf(sqrt(zeta) r)/r`; and point nuclei with fractional charge from
`PTR_FRAC_CHARGE`. Integer charges use `abs(CHARGE_OF)`, as in libcint. Zero
charge is a ghost; Gaussian `zeta=0` has libcint's point-nucleus meaning.
Negative/nonfinite zeta, unknown nuclear models and invalid offsets are rejected.
ECP contributions, spinors, derivative operators, periodic sums and nonzero
`PTR_RANGE_OMEGA` are outside this API. The Python nuclear wrapper rejects ECP
molecules; raw C callers must not substitute this operator for ECP integrals.

Check every C status: negative means failure. Validation failures leave output
untouched; any failure after computation starts clears the entire requested
output, including earlier batch elements. Python raises instead of returning
failed output. Finite input is not a guarantee that every extreme parameter is
numerically supported: the complex Rys panel budget and conditioning checks may
return `GPW_RANGE`/`GPW_NUMERIC`. There is no alternative integral backend or
approximate screening; exact zero contraction coefficients may be skipped.

## One-electron derivation

For a Cartesian primitive `(r-A)^l exp(-p|r-A|²)`, define
`P=A+i k/(2p)` and `phase=i k·A-k²/(4p)`. Overlap is
`(π/p)^(3/2) exp(phase) × product_d M_ld(i k_d/(2p), 1/(2p))`, where
`M_0=1`, `M_1=m`, `M_j=m M_(j-1)+(j-1)v M_(j-2)`.
Kinetic follows exactly from `-∇² exp(i k·r)/2 = k² exp(i k·r)/2`.

For a nucleus C, let `tau²=1` for point nuclei, or `zeta/(p+zeta)` for positive
Gaussian zeta; `D=P-C`, `T=p tau² D·D`, with an analytic dot product (no
conjugation). The Laplace representation gives

```text
-Z (2π/p) tau exp(phase) ∫_0^1 exp(-T u²)
    × product_d M_ld(P_d-A_d-tau² u² D_d, (1-tau² u²)/(2p)) du.
```

The polynomial has degree at most `l` in `u²`, so `floor(l/2)+1` Rys roots
integrate it. This calls the same complex `gpw_rule` already used by the
two-electron integrals. Weights include `exp(sigma)`, `sigma=min(Re(T),0)`;
the prefactor includes `exp(phase-sigma)`. For negative `Re(T)` its real exponent
is evaluated as `-p tau² |A-C|²-(1-tau²)k²/(4p)`, avoiding cancellation and
preserving point-nucleus tails when overlap underflows.

For exactly real nonnegative `T`, GPW calls the original `CINTrys_roots`, converts
its `u=s/(1-s)` nodes to `s`, and checks the reconstructed Boys moments. A rule
that fails this check is evaluated through the general GPW Rys construction.
General complex `T` uses scaled discrete Stieltjes polynomials and bilinear
products, with internal small-matrix QR and specialized one/two-root paths.
Both paths keep their numerical residual checks. Low-angular-momentum
recurrences use fixed loop bounds, and contiguous contraction/transform loops
allow compiler vectorization without changing the public ABI. Optimizations
must preserve numerical checks and use general mathematical structure, never
molecule-specific branches, fitted corrections, or relaxed acceptance tolerances.
Root reconstruction checks do not provide a uniform bound on discretization error;
the independent checks below establish accuracy at their tested inputs.

## Validation

```sh
cmake -S . -B build -DWITH_GPW=ON -DENABLE_TEST=ON \
  -DPYTHON_EXECUTABLE="$(command -v python)" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
python testsuite/gpw/test_int1e.py --library build/libgpw.so \
  --report build/gpw_int1e_validation.json
```

The independent Python check additionally needs `mpmath`; CMake registers it
as `gpw_int1e` when these Python dependencies are available. It compares Fourier
derivatives and explicit high-precision Boys moments, a direct Laplace integral,
and unmodified PySCF overlap/nuclear integrals at `k=0` (exact Gaussian exponent
splitting, without a small-exponent approximation). It covers all supported
angular momenta, three nuclear models, general contractions, spherical
transforms, phases, zero/large wavevectors, diffuse/tight exponents, batching,
normalizations, shared opt, stale inputs and failure semantics. Comparison
tolerance is `1e-11 + 1e-10*abs(reference)`, not bitwise equality to another
algorithm. The C smoke check exercises all ten entries and allocation bounds;
it can also be run in an AddressSanitizer/UndefinedBehaviorSanitizer build.

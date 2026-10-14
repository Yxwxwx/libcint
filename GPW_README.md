# Standalone mixed GTO–PW integrals

This libcint fork provides `libgpw` and `include/gpw.h`. It computes GTO–PW
overlap, kinetic, nuclear attraction and dipole, and the mixed two-electron `gpgg` and
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
export OMP_NUM_THREADS=8
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
    D = gpw.dipole(mol, k, opt=opt)     # (nk, 3, nao), x/y/z position, no -e
    H = T + V
    eri3 = gpw.gpgg(mol, k, opt=opt)    # (nk, nao, nao, nao)
    eri2 = gpw.gppg(mol, k, k, opt=opt) # (nk, nao, nao), zip pairs
    shell = gpw.nuc_by_shell(mol, (0,), k, opt=opt)
```

All six functions accept `cart=None` (use `mol.cart`), `cart=True` or
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
primitive-parameter table. One/two-root two-electron batches also reuse a
per-primitive, eight-k root/prefactor tile; `gpw_workspace_bytes` includes this storage.
The common plane-wave phase is computed once per k in each tile. Recurrence
reuses the prepared prefactor instead of recomputing T and its exponential.

OpenMP is enabled for `libgpw` when available; configure with
`-DGPW_WITH_OPENMP=OFF` for a serial build. It adds no BLAS/LAPACK dependency
and does not change the ordinary `libcint` target. Set `OMP_NUM_THREADS` before
starting Python (or use `omp_set_num_threads` from a C application). All five
integral types parallelize independent k points, using at most `nk` workers.
Calls inside an already active OpenMP region stay serial. For callers using
multiple processes, set the per-process thread count within the CPU allocation.

Each worker has private recurrence, contraction and spherical-transform
buffers. The prepared primitive table and opt are shared read-only. Worker
scratch is allocated per shell batch and freed before returning; it is not
included in the caller's `gpw_workspace_bytes`. Diagnostic counters include
all workers. Each integral keeps its contraction accumulation order and numerical
checks; a calculation failure clears the full output after all workers finish.
`OMP_WAIT_POLICY=PASSIVE` can avoid workers spinning between Python shell calls.

One/two-root two-electron batches schedule eight-k tiles. Within the existing
complex `|T| < 2` series domain, moment evaluation uses real/imaginary arrays
and OpenMP SIMD across k, without combining sums from different k points.
Scalar and SIMD paths share the root reconstruction and residual checks.
Real nonnegative T still uses libcint's native Rys routine. Outside the small
series domain, one/two-root batches with `|Re(T)| <= 20` and `|Im(T)| <= 12`
use the same 32-point quadrature as the general construction, with compensated
double moment sums across k. Cancellation amplification above 1000, or a failed
root reconstruction, returns to the original long-double moment construction.
Root reconstruction itself retains its long-double safeguards. Known
positive quadrature amplitudes supply weight moduli directly, avoiding
redundant long-double complex absolute values. These choices depend on root
count and numerical domain, never on molecular identity.

When the GNU compiler and system math library provide the vector exp/sin/cos
ABI, CMake enables it for `gpw_rys.c`. This vectorizes quadrature weights both
across k and across quadrature nodes. Otherwise the same formulas use scalar
math functions. No fast-math is enabled; only math errno side effects are
disabled in this source file, while all GPW value/status checks remain active.

The default build remains portable. On the deployment machine, the existing
`-DBUILD_MARCH_NATIVE=ON` option enables its wider SIMD instructions. The
benchmarked native build also uses `-DCMAKE_C_FLAGS=-ffp-contract=off` to avoid
FMA changing accumulation rounding; it does not enable fast-math. Native
binaries must not be moved to machines lacking those CPU instructions.

Full-AO Python `gpgg` computes only shell pairs `b >= c` and fills the other
half using `(G_a P_k|G_b G_c) = (G_a P_k|G_c G_b)` without conjugation. The
returned tensor is still full-sized; diagonal shell blocks are evaluated in
full. Explicit `shls` calls compute exactly the requested block. This symmetry
is not applied to `gppg` with general paired wavevectors.

## Workload and optimization target

The primary workload is a small Gaussian AO basis with many plane-wave k
points, potentially of order 100,000 or more. Small AO count does not restrict
angular momentum or contraction patterns. The current deliverable remains the
complete `gpgg` tensor `(nk, nao, nao, nao)`, including symmetry-related entries.
Internal k tiling may organize its evaluation, but an MO-only result, a direct
contraction, or a packed tensor does not replace that output requirement.

Optimization should prioritize the work repeated for each k: complex Rys
roots/weights, recurrence, root summation, contraction, spherical transformation
and writing the full tensor. Opt preparation and allocation improvements are
supporting measures; their importance must be measured on this workload.

Use libcint's existing Rys implementation as the engineering reference:

- `src/rys_roots.c`: root-count specialization and parameter-domain dispatch.
  Reuse its real nonnegative-T routines where valid; complex-T fast paths
  require their own stability analysis and independent numerical validation.
- `src/g2e.c`: shell-dependent compact recurrence dimensions, recurrence-base
  selection, precomputed Cartesian indices and unrolled low-order kernels.
- `src/cint2e.c`: root-count-specialized `CINTgout2e`, staged contraction,
  single-contraction buffer aliasing and exact-zero coefficient handling.

GPW already uses some of these ideas; further work should measure the remaining
gaps and adapt the same structure to complex arithmetic and batches of k.
Across-k SIMD is a GPW-specific extension of this approach. Preserve supported
inputs, numerical checks and error tolerances; use no molecule-specific tuning
or unvalidated screening. Benchmark the complete returned tensor and validate
all entries, separating kernel throughput from preparation and output costs.

## Definitions and normalization

Coordinates are bohr and wavevectors are bohr⁻¹. The core uses the bare plane
wave `P_k(r)=exp(+i k·r)`. Real GTOs follow libcint/PySCF normalization and AO
ordering, including the common s/p factors and all contraction columns.

| Entry | Integral |
| --- | --- |
| `ovlp` | `S_ak = <G_a | P_k>` |
| `kin` | `T_ak = <G_a | -∇²/2 | P_k> = (k²/2) S_ak` |
| `nuc` | `V_ak = <G_a | -Σ_A Z_A/r_A | P_k>` for point nuclei |
| `dipole` | `D_dak = <G_a | (r-O)_d | P_k>`, d = x, y, z; no electron-charge factor |
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

The eight single-wavevector symbols are
`gpw_int1e_{ovlp,kin,nuc,dipole}_gp_{cart,sph}`. `gpw_int1e_batch` selects `GPW_OVLP`,
`GPW_KIN`, `GPW_NUC` or `GPW_DIPOLE`. All take one real GTO shell index; there is no dummy PW
shell. Existing `gpw_int2e_*` signatures are unchanged. See `include/gpw.h` and
the executable C example/check `testsuite/gpw/smoke.c` for complete declarations.

Indices are `int32_t`; outputs are interleaved real/imaginary doubles. Capacity
is measured in complex numbers. Within an AO axis, contraction precedes the
Cartesian/spherical component. Batches are outermost. A workspace needs at
least the Cartesian shell element count, including all contractions, even for
spherical output. `opt=NULL` and `work=NULL` are valid.

Dipole returns `(3, nao)` for a single Python wavevector `(3,)`, or
`(nk, 3, nao)` for a batch `(nk, 3)`, including empty batches. Component order
is x, y, z; `dipole_by_shell(mol, (a,), k)` replaces `nao` by the shell's AO
count. The C batch layout is also `(count, 3, nAO_shell)`, and its workspace
requires `3*ncart_shell` complex elements. Scalar operators keep their original
shapes and workspace requirements. Dipole is an additive ABI-1 extension.

The operator is the position relative to the common origin, without the
electronic charge (-1), in atomic units. The origin is read from libcint's
`env[PTR_COMMON_ORIG:PTR_COMMON_ORIG+3]`, as for PySCF `int1e_r`; it defaults
to zero. For example:

```python
with mol.with_common_orig((0.2, -0.1, 0.3)):  # bohr
    with gpw.prepare(mol) as opt:  # create after setting the origin
        D = gpw.dipole(mol, k, opt=opt)  # (nk, 3, nao) for batched k
```

Changing the origin invalidates an existing OPT, just like other changes to
`mol._env`. For bare plane waves, `D_d = -i*dS/dk_d - O_d*S`. The implementation
uses analytic Gaussian moments through order `l+1`; it never uses numerical
differentiation. All four plane-wave normalization options apply as for overlap.

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

For complex T with `a=Re(T)>40`, the large-parameter constants in libcint's
`roots_for_x0.dat` also give an analytic infinite-interval rule:
`s_i=POLY_LARGEX_RT_i/T`, `w_i=POLY_LARGEX_WW_i*sqrt(pi)/(2*sqrt(T))`.
This is used only when the omitted tail is bounded. For the highest moment
`M=2*n-1`, writing `s=u²` and using `log(s)<=s-1` for `s>=1` gives

```text
|tail_M| <= exp(-a)/(2*(a-M+1/2)),
|tail_M| / |Gamma(M+1/2)/(2*T^(M+1/2))|
    <= exp(-a)*|T|^(M+1/2) / ((a-M+1/2)*Gamma(M+1/2)).
```

For `n<=10`, `a>40`, this bounds all lower relative moment tails as well.
The implementation requires this bound to be at most `4*DBL_EPSILON` and
also checks reconstructed scaled moments. Otherwise it uses the general
construction. Thus real-domain fit polynomials are not extrapolated to complex
arguments; only the asymptotic constants are reused with a complex-domain bound.

The parent GPW workspace's `tests/gpw_domain/check.py` checks this dispatch,
SIMD batch boundaries, cancellation near a zero of the zeroth moment, and
independent 80-digit s/s, p/p and third-derivative p/d references. Its
`debug/benchmark_gpw_domain.py` alternates complete old/new H2O/cc-pVDZ tensors
on four threads with 2500 k points. Both are Python entry points.

## Validation

The dipole check uses direct real-space Gauss–Hermite quadrature of PySCF AO
values on two grid orders, plus the Fourier derivative and origin-shift
identities. It covers H2O/Ne cc-pVDZ, all supported angular momenta, general
contractions, Cartesian/spherical AOs, normalization, 1/2/4 OpenMP threads,
empty/single/batched k, both native single-k entries, capacity checks, stale
OPT and failure output handling. Run it from the libcint directory:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python testsuite/gpw/test_dipole.py \
  --library build/libgpw.so --report build/gpw_dipole_validation.json
```

CMake registers this Python check as `gpw_dipole` with `ENABLE_TEST=ON` when
the listed Python test dependencies are installed.

The Python regression for OpenMP and symmetry can also be run independently:

```sh
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 python testsuite/test_gpw_parallel.py \
  --library build/libgpw.so --report build/gpw_parallel_validation.json
```

It compares 1/2/4/8 threads, full tensors against independently evaluated shell
blocks, all five operators, Cartesian/spherical AOs, general contractions,
high angular momentum, empty/single/batched wavevectors, workspaces, counters
and failure handling. For a build without OpenMP, add `--serial`; an older
library may be supplied with `--reference` to check numerical compatibility.
The regression also checks SIMD tile boundaries and mixed Rys parameter domains.
An independent check of all ten root counts against 80-digit moments is:

```sh
python testsuite/test_gpw_roots.py --library build/libgpw.so \
  --report build/gpw_roots_validation.json
```

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

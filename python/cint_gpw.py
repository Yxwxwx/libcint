"""NumPy/PySCF bindings for the mixed GTO–plane-wave integrals in libgpw.

Coordinates are in bohr, wavevectors in bohr^-1, and P_k(r) = exp(+i k·r).
All integral results are complex128. Passing one wavevector of shape (3,)
returns no batch axis; passing (nk, 3) adds one leading batch axis. For
gppg, that axis counts (k[i], kp[i]) pairs, not independent k and kp axes.
The remaining axes always count Gaussian AOs: three for gpgg, two for gppg,
and one for each one-electron integral.

Common keyword arguments for the five integral functions:
    shls: Zero-based GTO shell indices, one per Gaussian AO axis. Omit to
        return all AO combinations.
    cart: Use Cartesian AOs if true, real spherical AOs if false; None uses
        mol.cart.
    pw_norm: 'bare' (default), 'delta3', 'box', or 'legacy_k'. The factor is
        applied once per plane wave, so gppg receives two factors.
    volume: Positive box volume in bohr^3, required only for pw_norm='box'.
    opt: An open GPW Prepared cache from prepare(mol), or None.

The bindings match libcint/include/gpw.h, ABI 1. The second plane wave in
gppg is conjugated internally; pass its actual wavevector.
"""
import ctypes as ct
from functools import lru_cache
import os
import itertools
import weakref
from ctypes.util import find_library
from pathlib import Path
import numpy as np

PInt = ct.POINTER(ct.c_int32)
PDouble = ct.POINTER(ct.c_double)
Handle = ct.c_void_p
_env_args = [PInt, ct.c_int32, PInt, ct.c_int32, PDouble, ct.c_size_t]


@lru_cache(maxsize=1)
def library():
    """Load and cache the ABI-1 libgpw shared library and C signatures.

    GPW_LIBRARY, if set before the first call, must be an absolute path.
    Otherwise the source-tree build and then the system library are tried.

    Returns:
        A ctypes CDLL with GPW function signatures configured.

    Raises:
        ValueError: GPW_LIBRARY is a relative path.
        OSError: The shared library cannot be loaded.
        ImportError: The library has an incompatible ABI.
    """
    path = os.environ.get('GPW_LIBRARY')
    if path is not None and not Path(path).is_absolute():
        raise ValueError('GPW_LIBRARY must be an absolute path')
    local = Path(__file__).resolve().parents[1]/'build/libgpw.so'
    lib = ct.CDLL(path or (str(local) if local.is_file() else
                          find_library('gpw') or 'libgpw.so'), mode=ct.RTLD_LOCAL)
    signatures = {
        'gpw_abi_version': ([], ct.c_int),
        'gpw_integer_bytes': ([], ct.c_int),
        'gpw_build_version': ([], ct.c_char_p),
        'gpw_error_string': ([ct.c_int], ct.c_char_p),
        'gpw_opt_create': ([ct.POINTER(Handle)] + _env_args, ct.c_int),
        'gpw_opt_destroy': ([Handle], None),
        'gpw_workspace_create': ([ct.POINTER(Handle), ct.c_size_t], ct.c_int),
        'gpw_workspace_destroy': ([Handle], None),
        'gpw_workspace_bytes': ([Handle], ct.c_size_t),
        'gpw_workspace_rules': ([Handle], ct.c_uint64),
        'gpw_workspace_special': ([Handle], ct.c_uint64),
        'gpw_int1e_batch': ([PDouble, ct.c_size_t, ct.c_int32, ct.c_int32, PInt,
                            PDouble, ct.c_size_t] + _env_args + [Handle, Handle], ct.c_int),
        'gpw_int2e_batch': ([PDouble, ct.c_size_t, ct.c_int32, ct.c_int32, PInt,
                            PDouble, PDouble, ct.c_size_t] + _env_args + [Handle, Handle], ct.c_int),
        'gpw_rys_rule': ([ct.c_int32, ct.c_double, ct.c_double, PDouble, PDouble,
                          PDouble, PDouble], ct.c_int),
        'gpw_boys_scaled': ([ct.c_int32, ct.c_double, ct.c_double, PDouble], ct.c_int),
    }
    for kind in ['gpgg', 'gppg']:
        for rep in ['cart', 'sph']:
            args = [PDouble, ct.c_size_t, PInt, PDouble]
            if kind == 'gppg':
                args.append(PDouble)
            signatures[f'gpw_int2e_{kind}_{rep}'] = (args + _env_args + [Handle, Handle], ct.c_int)
    for op in ('ovlp', 'kin', 'nuc'):
        for rep in ('cart', 'sph'):
            signatures[f'gpw_int1e_{op}_gp_{rep}'] = (
                [PDouble, ct.c_size_t, PInt, PDouble] + _env_args + [Handle, Handle], ct.c_int)
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    if lib.gpw_abi_version() != 1 or lib.gpw_integer_bytes() != 4:
        raise ImportError('GPW ABI 1 with int32 indices is required')
    return lib


def double_pointer(array):
    """Return a ctypes double pointer to an array, or NULL for None."""
    return None if array is None else array.ctypes.data_as(PDouble)


def int_pointer(array):
    """Return a ctypes int32 pointer to an array's existing data."""
    return array.ctypes.data_as(PInt)


def inputs(atm, bas, env):
    """Convert PySCF molecule arrays to contiguous GPW ABI inputs.

    Args:
        atm: Atom table with shape (natm, 6) and integer entries.
        bas: Shell table with shape (nbas, 8) and integer entries.
        env: One-dimensional real-valued environment array.

    Returns:
        A pair ``(arrays, c_args)``. Keep ``arrays`` alive while passing
        ``c_args`` to C; it owns the buffers referenced by the pointers.

    Raises:
        ValueError: An array has an invalid shape, type, or int32 range.
    """
    for value, cols in [(atm, 6), (bas, 8)]:
        if value.ndim != 2 or value.shape[1] != cols or value.dtype.kind not in 'iu':
            raise ValueError('invalid atm/bas shape or integer dtype')
        if value.dtype != np.int32 and value.size and (value.max() > np.iinfo(np.int32).max or value.min() < np.iinfo(np.int32).min):
            raise ValueError('atm/bas exceed int32 range')
        if value.shape[0] > np.iinfo(np.int32).max:
            raise ValueError('too many atoms or shells')
    if env.ndim != 1 or np.iscomplexobj(env):
        raise ValueError('env must be a real vector')
    arrays = (np.ascontiguousarray(atm, dtype=np.int32),
              np.ascontiguousarray(bas, dtype=np.int32),
              np.ascontiguousarray(env, dtype=np.float64))
    a, b, e = arrays
    return arrays, (int_pointer(a), len(a), int_pointer(b), len(b), double_pointer(e), len(e))


def check(status, context):
    """Raise RuntimeError with context for a negative GPW status code."""
    if status < 0:
        message = library().gpw_error_string(status).decode()
        raise RuntimeError(f'GPW {message}; {context}')


class Prepared:
    """Immutable GPW-specific cache; share only while it remains open."""
    def __init__(self, mol):
        """Create a cache from the current PySCF molecule arrays."""
        arrays, args = inputs(mol._atm, mol._bas, mol._env)
        lib = library()
        handle = Handle()
        check(lib.gpw_opt_create(ct.byref(handle), *args), 'preparing molecule')
        self._handle = handle
        self._finalizer = weakref.finalize(self, lib.gpw_opt_destroy, handle)

    def close(self):
        """Release the native cache; repeated calls are harmless."""
        self._finalizer()
        self._handle = None

    def __enter__(self):
        """Return this open cache for use in a ``with`` statement."""
        if self._handle is None:
            raise ValueError('prepared object is closed')
        return self

    def __exit__(self, *exc):
        """Close the cache on context exit without suppressing exceptions."""
        self.close()


def prepare(mol):
    """Create a reusable GPW cache from a PySCF molecule.

    Args:
        mol: PySCF Mole providing ``_atm``, ``_bas``, and ``_env``.

    Returns:
        A context-managed Prepared object for ``opt=`` calls. GPW rejects
        later changes to the molecule arrays; this is not libcint's CINTOpt.
    """
    return Prepared(mol)


def _waves(k):
    """Return contiguous real ``(nk, 3)`` wavevectors and a single-input flag.

    Args:
        k: One wavevector ``(3,)`` or a batch ``(nk, 3)`` in bohr^-1.

    Raises:
        ValueError: The wavevectors are complex, nonfinite, or mis-shaped.
    """
    k = np.asarray(k)
    if np.iscomplexobj(k):
        raise ValueError('wavevectors must be real')
    single = k.shape == (3,)
    if not single and (k.ndim != 2 or k.shape[1] != 3):
        raise ValueError('wavevectors must have shape (3,) or (n,3)')
    k = np.ascontiguousarray(k, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(k).all():
        raise ValueError('wavevectors must be finite')
    return k, single


def _normalization(k, pw_norm, volume):
    """Return one real normalization factor per wavevector in ``k``.

    ``bare`` gives 1, ``delta3`` gives (2π)^(-3/2), ``box`` gives
    volume^(-1/2), and ``legacy_k`` gives (2π)^(-3/2)|k|.

    Raises:
        ValueError: The mode is unknown or box volume is not positive finite.
    """
    if pw_norm == 'bare':
        return np.ones(len(k))
    if pw_norm == 'delta3':
        return np.full(len(k), (2*np.pi)**-1.5)
    if pw_norm == 'legacy_k':
        return (2*np.pi)**-1.5 * np.linalg.norm(k, axis=1)
    if pw_norm == 'box':
        if volume is None or np.ndim(volume) or np.iscomplexobj(volume) or not np.isfinite(volume) or volume <= 0:
            raise ValueError('box normalization requires positive finite volume in bohr^3')
        return np.full(len(k), volume**-.5)
    raise ValueError('pw_norm must be bare, delta3, box, or legacy_k')


def _evaluate(kind, mol, k, kp=None, *, shls=None, cart=None,
              pw_norm='bare', volume=None, opt=None):
    """Evaluate one integral kind and assemble its AO shell blocks.

    ``kind`` is ovlp, kin, nuc, gpgg, or gppg. For gppg, ``k`` and ``kp``
    must have identical shapes and are paired by position. ``shls`` selects
    one shell per AO axis; other options follow the module conventions.

    Returns:
        A complex128 AO array, with a leading batch axis only for batched
        wavevectors. The number of AO axes is one, three, or two for the
        one-electron, gpgg, and gppg integrals, respectively.

    Raises:
        ValueError: Invalid options, wavevectors, shells, or unsupported ECP.
        RuntimeError: The native GPW call fails or ``opt`` is stale.
    """
    rank = 3 if kind == 'gpgg' else 2 if kind == 'gppg' else 1
    waves, single = _waves(k)
    other = None
    if rank == 2:
        other, other_single = _waves(kp)
        if other.shape != waves.shape or other_single != single:
            raise ValueError('k and kp must have exactly the same shape (zip pairing)')
    cart = mol.cart if cart is None else cart
    if not isinstance(cart, (bool, np.bool_)):
        raise ValueError('cart must be True, False, or None')
    if opt is not None and (not isinstance(opt, Prepared) or opt._handle is None):
        raise ValueError('opt must be an open Prepared object')
    if kind == 'nuc' and np.size(getattr(mol, '_ecpbas', [])):
        raise ValueError('GPW nuclear attraction does not include ECP integrals')
    arrays, args = inputs(mol._atm, mol._bas, mol._env)
    atm, bas, env = arrays
    # Validate before sizing/allocating output, including empty batches.
    if np.any((bas[:, 1] < 0) | (bas[:, 1] > 6) | (bas[:, 3] < 1)):
        raise ValueError('GPW requires l=0..6 and positive NCTR_OF')
    sizes = np.array([((int(l)+1)*(int(l)+2)//2 if cart else 2*int(l)+1)*int(nc)
                      for l, nc in bas[:, [1, 3]]], dtype=np.int64)
    if shls is not None:
        shls = np.asarray(shls)
        if shls.shape != (rank,) or shls.dtype.kind not in 'iu' or np.any(shls < 0) or np.any(shls >= len(bas)):
            raise ValueError(f'shls must contain {rank} valid integer shell indices')
        selected = [tuple(int(s) for s in shls)]
        shape = tuple(int(sizes[s]) for s in shls)
    else:
        selected = itertools.product(range(len(bas)), repeat=rank)
        shape = (int(sizes.sum()),)*rank
    scale = _normalization(waves, pw_norm, volume)
    if other is not None:
        scale *= _normalization(other, pw_norm, volume)
    out = np.empty((len(waves), *shape), dtype=np.complex128)
    loc = np.concatenate(([0], np.cumsum(sizes)))
    lib = library()
    handle = None if opt is None else opt._handle
    cart_sizes = [(int(l)+1)*(int(l)+2)//2*int(nc) for l, nc in bas[:, [1, 3]]]
    capacity = 1
    for size in ([cart_sizes[s] for s in shls] if shls is not None else
                 [max(cart_sizes, default=0)]*rank):
        capacity *= size
    workspace = Handle()
    check(lib.gpw_workspace_create(ct.byref(workspace), capacity), 'allocating workspace')
    try:
        for shell in selected:
            indices = np.asarray(shell, dtype=np.int32)
            block = np.empty((len(waves), *(int(sizes[s]) for s in shell)), dtype=np.complex128)
            common = (double_pointer(block), block.size)
            if rank == 1:
                status = lib.gpw_int1e_batch(*common, {'ovlp': 1, 'kin': 2, 'nuc': 3}[kind], int(cart),
                    int_pointer(indices), double_pointer(waves), len(waves), *args, handle, workspace)
            else:
                status = lib.gpw_int2e_batch(*common, rank, int(cart), int_pointer(indices),
                    double_pointer(waves), double_pointer(other), len(waves), *args, handle, workspace)
            check(status, f'{kind}, shls={shell}')
            if shls is None:
                out[(slice(None),) + tuple(slice(loc[s], loc[s+1]) for s in shell)] = block
            else:
                out[...] = block
    finally:
        lib.gpw_workspace_destroy(workspace)
    out *= scale.reshape((-1,) + (1,)*rank)
    return out[0] if single else out


def ovlp(mol, k, **kwargs):
    """Compute the Gaussian–plane-wave overlap ``<G_a|P_k>``.

    Args:
        mol: PySCF Mole containing the Gaussian AO basis.
        k: One real wavevector ``(3,)`` or a batch ``(nk, 3)`` in bohr^-1.
        **kwargs: Common ``shls``, ``cart``, ``pw_norm``, ``volume``, and
            ``opt`` options documented at module level.

    Returns:
        Complex AO overlaps with shape ``(nao,)`` or ``(nk, nao)``. If
        ``shls`` is given, ``nao`` is the selected shell's AO count.
    """
    return _evaluate('ovlp', mol, k, **kwargs)


def kin(mol, k, **kwargs):
    """Compute ``<G_a|-∇²/2|P_k>`` with the operator acting on ``P_k``.

    Args:
        mol: PySCF Mole containing the Gaussian AO basis.
        k: One real wavevector ``(3,)`` or a batch ``(nk, 3)`` in bohr^-1.
        **kwargs: Common options documented at module level.

    Returns:
        Complex array shaped like :func:`ovlp`. Each element equals its
        overlap counterpart times ``|k|²/2``.
    """
    return _evaluate('kin', mol, k, **kwargs)


def nuc(mol, k, **kwargs):
    """Compute ``<G_a|V_nuc|P_k>`` for the molecule's nuclei.

    Point, Gaussian-charge, and fractional-charge nuclei are supported;
    ECP contributions are not included.

    Args:
        mol: PySCF Mole containing the Gaussian AO basis and nuclei.
        k: One real wavevector ``(3,)`` or a batch ``(nk, 3)`` in bohr^-1.
        **kwargs: Common options documented at module level.

    Returns:
        Complex array shaped like :func:`ovlp`.

    Raises:
        ValueError: The molecule contains ECP shells.
    """
    return _evaluate('nuc', mol, k, **kwargs)


def gpgg(mol, k, **kwargs):
    """Compute the chemists' two-electron integral ``(G_a P_k|G_b G_c)``.

    Args:
        mol: PySCF Mole containing the Gaussian AO basis.
        k: One real wavevector ``(3,)`` or a batch ``(nk, 3)`` in bohr^-1.
        **kwargs: Common options documented at module level; ``shls``
            selects three Gaussian shells in ``(a, b, c)`` order.

    Returns:
        Complex array ``(nao, nao, nao)`` or ``(nk, nao, nao, nao)``.
        The three AO axes index ``(a, b, c)``; the leading axis, if present,
        indexes ``k[i]``. Shell selection replaces each ``nao`` with that
        shell's AO count.
    """
    return _evaluate('gpgg', mol, k, **kwargs)


def gppg(mol, k, kp, **kwargs):
    """Compute the chemists' integral ``(G_a P_k|P_kp G_b)``.

    The second plane wave is conjugated internally. Batched wavevectors are
    paired by position: result ``[i, a, b]`` is
    ``(G_a P_{k[i]}|P_{kp[i]} G_b)``. There is one batch axis for the pairs,
    not one axis per plane wave; no Cartesian product is made.

    Args:
        mol: PySCF Mole containing the Gaussian AO basis.
        k: One real wavevector ``(3,)`` or a batch ``(nk, 3)`` in bohr^-1.
        kp: Actual wavevector(s) for the second plane wave, with exactly the
            same shape as ``k``.
        **kwargs: Common options documented at module level; ``shls``
            selects two Gaussian shells in ``(a, b)`` order.

    Returns:
        Complex array ``(nao, nao)`` for two single ``(3,)`` inputs, or
        ``(nk, nao, nao)`` for two ``(nk, 3)`` inputs. The AO axes index
        ``(a, b)``. Shell selection replaces each ``nao`` with that shell's
        AO count.

    Example:
        If ``k.shape == kp.shape == (2, 3)`` and ``nao == 7``, then
        ``gppg(mol, k, kp).shape == (2, 7, 7)``: only pairs ``(k[0], kp[0])``
        and ``(k[1], kp[1])`` are computed. By contrast,
        ``gppg(mol, k[0], k[1]).shape == (7, 7)``. To get all four cross
        pairs with shape ``(2, 2, 7, 7)``::

            ki = np.repeat(k, len(kp), axis=0)
            kj = np.tile(kp, (len(k), 1))
            all_pairs = gppg(mol, ki, kj).reshape(len(k), len(kp), nao, nao)
    """
    return _evaluate('gppg', mol, k, kp, **kwargs)


def ovlp_by_shell(mol, shls, k, **kwargs):
    """Compute ``<G_a|P_k>`` for the one zero-based GTO shell in ``shls``.

    Returns a complex array ``(n_a,)`` or ``(nk, n_a)``; see :func:`ovlp`
    for ``mol``, ``k``, and the remaining keyword options.
    """
    return ovlp(mol, k, shls=shls, **kwargs)


def kin_by_shell(mol, shls, k, **kwargs):
    """Compute ``<G_a|-∇²/2|P_k>`` for the shell in ``shls``.

    Returns a complex array ``(n_a,)`` or ``(nk, n_a)``; see :func:`kin`
    for ``mol``, ``k``, and the remaining keyword options.
    """
    return kin(mol, k, shls=shls, **kwargs)


def nuc_by_shell(mol, shls, k, **kwargs):
    """Compute ``<G_a|V_nuc|P_k>`` for the shell in ``shls``.

    Returns a complex array ``(n_a,)`` or ``(nk, n_a)``; see :func:`nuc`
    for ``mol``, ``k``, and the remaining keyword options.
    """
    return nuc(mol, k, shls=shls, **kwargs)


def gpgg_by_shell(mol, shls, k, **kwargs):
    """Compute ``(G_a P_k|G_b G_c)`` for ``shls=(a, b, c)``.

    Returns ``(n_a, n_b, n_c)`` or ``(nk, n_a, n_b, n_c)``; see
    :func:`gpgg` for ``mol``, ``k``, and the remaining keyword options.
    """
    return gpgg(mol, k, shls=shls, **kwargs)


def gppg_by_shell(mol, shls, k, kp, **kwargs):
    """Compute ``(G_a P_k|P_kp G_b)`` for ``shls=(a, b)``.

    Returns ``(n_a, n_b)`` or ``(nk, n_a, n_b)``; see :func:`gppg` for
    ``mol``, ``k``, ``kp``, and the remaining keyword options.
    """
    return gppg(mol, k, kp, shls=shls, **kwargs)

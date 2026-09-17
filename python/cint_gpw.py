"""ctypes declarations matching libcint/include/gpw.h (ABI 1)."""
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
    return None if array is None else array.ctypes.data_as(PDouble)


def int_pointer(array):
    return array.ctypes.data_as(PInt)


def inputs(atm, bas, env):
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
    if status < 0:
        message = library().gpw_error_string(status).decode()
        raise RuntimeError(f'GPW {message}; {context}')


class Prepared:
    """Immutable GPWOpt; share between threads, close only after calls finish."""
    def __init__(self, mol):
        arrays, args = inputs(mol._atm, mol._bas, mol._env)
        lib = library()
        handle = Handle()
        check(lib.gpw_opt_create(ct.byref(handle), *args), 'preparing molecule')
        self._handle = handle
        self._finalizer = weakref.finalize(self, lib.gpw_opt_destroy, handle)

    def close(self):
        self._finalizer()
        self._handle = None

    def __enter__(self):
        if self._handle is None:
            raise ValueError('prepared object is closed')
        return self

    def __exit__(self, *exc):
        self.close()


def prepare(mol):
    """Snapshot atm/bas/env for opt= calls. Changed inputs raise an error."""
    return Prepared(mol)


def _waves(k):
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
    for shell in selected:
        indices = np.asarray(shell, dtype=np.int32)
        block = np.empty((len(waves), *(int(sizes[s]) for s in shell)), dtype=np.complex128)
        common = (double_pointer(block), block.size)
        if rank == 1:
            status = lib.gpw_int1e_batch(*common, {'ovlp': 1, 'kin': 2, 'nuc': 3}[kind], int(cart),
                int_pointer(indices), double_pointer(waves), len(waves), *args, handle, None)
        else:
            status = lib.gpw_int2e_batch(*common, rank, int(cart), int_pointer(indices),
                double_pointer(waves), double_pointer(other), len(waves), *args, handle, None)
        check(status, f'{kind}, shls={shell}')
        if shls is None:
            out[(slice(None),) + tuple(slice(loc[s], loc[s+1]) for s in shell)] = block
        else:
            out[...] = block
    out *= scale.reshape((-1,) + (1,)*rank)
    return out[0] if single else out


def ovlp(mol, k, **kwargs):
    """<G|P>: (nao,) or (nk,nao); options: shls, cart, pw_norm, volume, opt."""
    return _evaluate('ovlp', mol, k, **kwargs)


def kin(mol, k, **kwargs):
    """<G|-laplacian/2|P>, with the operator acting on exp(+i k.r)."""
    return _evaluate('kin', mol, k, **kwargs)


def nuc(mol, k, **kwargs):
    """<G|-sum_A Z_A/r_A|P>, including finite Gaussian/fractional nuclei."""
    return _evaluate('nuc', mol, k, **kwargs)


def gpgg(mol, k, **kwargs):
    """Chemists' (G_a P_k|G_b G_c): (nao,nao,nao) or (nk,nao,nao,nao)."""
    return _evaluate('gpgg', mol, k, **kwargs)


def gppg(mol, k, kp, **kwargs):
    """Chemists' (G_a P_k|P_kp G_b): (nao,nao) or (nk,nao,nao); zip pairs."""
    return _evaluate('gppg', mol, k, kp, **kwargs)


def ovlp_by_shell(mol, shls, k, **kwargs):
    return ovlp(mol, k, shls=shls, **kwargs)


def kin_by_shell(mol, shls, k, **kwargs):
    return kin(mol, k, shls=shls, **kwargs)


def nuc_by_shell(mol, shls, k, **kwargs):
    return nuc(mol, k, shls=shls, **kwargs)


def gpgg_by_shell(mol, shls, k, **kwargs):
    return gpgg(mol, k, shls=shls, **kwargs)


def gppg_by_shell(mol, shls, k, kp, **kwargs):
    return gppg(mol, k, kp, shls=shls, **kwargs)

"""
verify.py – Behavioural verification for the LL1 decomposition module.

Every check() call is annotated with the requirement it validates; the
requirement numbers refer to the original task specification reproduced
below as comments (REQ-n).

Original requirements:
  REQ-1  New module ll1 (unconstrained): tensor = sum_r A_r ⊗ c_r.
  REQ-2  constrained_LL1 class inherits from LL1, adds Stokes constraints.
  REQ-3  Stokes vectors satisfy  S0 >= ||(S1,S2,S3)||  and  S0 >= 0.
  REQ-4  Activation maps A_r are non-negative in the constrained model.
  REQ-5  check_ll1_uniqueness(shape, rank) returns True/False.
  REQ-6  ll1_als  – vanilla ALS for the unconstrained case.
  REQ-7  ll1_bpg  – Block-Proximal Gradient for the Stokes case.
  REQ-8  ll1_ao_admm – AO-ADMM for the Stokes scenario.
  REQ-9  gen_ll1 in tensorly.datasets generates LL1 tensors with optional
         Stokes constraints.
  REQ-10 Unconstrained noiseless convergence: rel error < 1e-9.
  REQ-11 Constrained noisy convergence (noise_level=1e-6): rel error < 1e-4.
  REQ-R  REGRESSION: existing CP decomposition must still work.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Partial-credit helpers (Rule 6, Rule 7, Rule 8)
# ---------------------------------------------------------------------------

CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label, passed, diag=""):
    """Run one assertion; print result; accumulate credit."""
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    status = "PASS" if passed else "FAIL"
    msg = f"[{status}] {label}"
    if diag:
        msg += f" | {diag}"
    print(msg)
    if passed:
        CHECKS_PASSED += 1


def write_reward_and_exit():
    """Write fractional reward.txt and always exit 0 (Rule 8)."""
    score = CHECKS_PASSED / CHECKS_TOTAL if CHECKS_TOTAL else 0.0
    reward_path = os.path.join(os.path.dirname(__file__), "reward.txt")
    with open(reward_path, "w") as fh:
        fh.write(f"{score:.4f}\n")
    print(
        f"\n{'='*60}\n"
        f"Result: {CHECKS_PASSED}/{CHECKS_TOTAL} checks passed  "
        f"(score = {score:.4f})\n"
        f"reward.txt written to {reward_path}\n"
        f"{'='*60}"
    )
    sys.exit(0)   # always 0 so test.sh does not overwrite reward.txt (Rule 8)


# ---------------------------------------------------------------------------
# Defensive imports (Rule 2)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(__file__)
sys.path.insert(0, _REPO_ROOT)

try:
    import numpy as np
    import tensorly as tl
except ImportError as exc:
    print(f"FATAL: Could not import numpy/tensorly from {_REPO_ROOT}: {exc}")
    sys.exit(1)

# --- decomposition module ---
try:
    from tensorly.decomposition import (
        ll1_als,
        ll1_bpg,
        ll1_ao_admm,
        check_ll1_uniqueness,
        initialize_ll1,
        LL1,
        ConstrainedLL1,
    )
except ImportError as exc:
    expected = os.path.join(_REPO_ROOT, "tensorly", "decomposition", "_ll1.py")
    print(
        f"FATAL: Could not import LL1 symbols from tensorly.decomposition.\n"
        f"  Expected implementation at: {expected}\n"
        f"  ImportError: {exc}"
    )
    sys.exit(1)

# --- datasets module ---
try:
    from tensorly.datasets import gen_ll1
except ImportError as exc:
    expected = os.path.join(_REPO_ROOT, "tensorly", "datasets", "synthetic.py")
    print(
        f"FATAL: Could not import gen_ll1 from tensorly.datasets.\n"
        f"  Expected implementation at: {expected}\n"
        f"  ImportError: {exc}"
    )
    sys.exit(1)

# --- internal helpers (imported for white-box reconstruction check) ---
try:
    from tensorly.decomposition._ll1 import _ll1_to_tensor, _proj_stokes_matrix
except ImportError as exc:
    print(f"WARNING: Could not import internal helpers: {exc}")
    _ll1_to_tensor = None
    _proj_stokes_matrix = None

# --- regression: existing CP decomposition ---
try:
    from tensorly.decomposition import parafac
    from tensorly.cp_tensor import cp_to_tensor
    _cp_available = True
except ImportError:
    _cp_available = False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _rel_error(T_true, T_est):
    num = float(tl.to_numpy(tl.norm(T_true - T_est)))
    den = float(tl.to_numpy(tl.norm(T_true)))
    return num / (den + 1e-14)


def _stokes_valid(vectors_np):
    """Return True if every column is a valid Stokes vector."""
    R = vectors_np.shape[1]
    for r in range(R):
        s0 = vectors_np[0, r]
        s_pol_n = np.linalg.norm(vectors_np[1:, r])
        if s0 < -1e-8 or s0 < s_pol_n - 1e-8:
            return False, r, s0, s_pol_n
    return True, -1, 0.0, 0.0


# ===========================================================================
# REQ-1  ll1_als exists and decomposes a tensor as sum of matrix⊗vector terms
# ===========================================================================

print("\n--- REQ-1: ll1_als – unconstrained LL1 model exists ---")

_rng = tl.check_random_state(0)
_T_small, _M_small, _V_small = gen_ll1((5, 4, 3), rank=2, random_state=0)

# Check that ll1_als is callable and returns (matrices, vectors)
try:
    _res = ll1_als(_T_small, rank=2, n_iter_max=10, random_state=0)
    _matrices, _vectors = _res
    check(
        "REQ-1a: ll1_als returns (matrices, vectors) tuple",
        isinstance(_matrices, list) and len(_matrices) == 2,
        diag=f"len(matrices)={len(_matrices)}"
    )
    check(
        "REQ-1b: each matrix has shape (I, J)",
        all(tl.shape(m) == (5, 4) for m in _matrices),
        diag=f"shapes={[tl.shape(m) for m in _matrices]}"
    )
    check(
        "REQ-1c: vectors matrix has shape (K, R)",
        tl.shape(_vectors) == (3, 2),
        diag=f"shape(vectors)={tl.shape(_vectors)}"
    )
except Exception as exc:
    check("REQ-1a: ll1_als callable", False, diag=str(exc))
    check("REQ-1b: matrix shapes",    False,
          diag="(skipped – exception above)")
    check("REQ-1c: vector shape",     False,
          diag="(skipped – exception above)")

# Check reconstruction: T ≈ sum_r A_r ⊗ c_r  (Rule 5: regression via exact data)
if _ll1_to_tensor is not None:
    try:
        _T_rec = _ll1_to_tensor(_matrices, _vectors)
        _shape_ok = tl.shape(_T_rec) == (5, 4, 3)
        check(
            "REQ-1d: reconstruction from factors has correct shape",
            _shape_ok,
            diag=f"got {tl.shape(_T_rec)}"
        )
        # Verify element-wise: T[i,j,k] = sum_r A_r[i,j]*c_r[k]
        _T_np = tl.to_numpy(_T_small)
        _TR_np = tl.to_numpy(_T_rec)
        _elem_ok = abs(_TR_np[0, 0, 0] - sum(
            tl.to_numpy(_matrices[r])[0, 0] * tl.to_numpy(_vectors)[0, r]
            for r in range(2)
        )) < 1e-10
        check(
            "REQ-1e: element-wise value matches sum_r A_r[i,j]*c_r[k]",
            _elem_ok,
            diag=f"diff={abs(_TR_np[0, 0, 0] - sum(tl.to_numpy(_matrices[r])[0, 0]*tl.to_numpy(_vectors)[0, r] for r in range(2))):.2e}"
        )
    except Exception as exc:
        check("REQ-1d: reconstruction shape", False, diag=str(exc))
        check("REQ-1e: element-wise value",   False, diag="(skipped)")

# ===========================================================================
# REQ-2  constrained_LL1 inherits from LL1
# ===========================================================================

print("\n--- REQ-2: ConstrainedLL1 inherits from LL1 ---")

check(
    "REQ-2a: ConstrainedLL1 is a subclass of LL1",
    issubclass(ConstrainedLL1, LL1),
    diag=f"MRO: {[c.__name__ for c in ConstrainedLL1.__mro__]}"
)

# fit_transform must return (matrices, vectors) without crashing
try:
    _T_s4, _, _ = gen_ll1((6, 5, 4), rank=2, stokes_constrained=True,
                          non_negative_matrices=True, random_state=1)
    _dec = ConstrainedLL1(rank=2, n_iter_max=30, method="bpg", random_state=1)
    _res2 = _dec.fit_transform(_T_s4)
    _m2, _v2 = _res2
    check(
        "REQ-2b: ConstrainedLL1.fit_transform returns (matrices, vectors)",
        isinstance(_m2, list) and len(_m2) == 2,
        diag=f"len(matrices)={len(_m2)}, shape(vectors)={tl.shape(_v2)}"
    )
    check(
        "REQ-2c: ConstrainedLL1 stores decomposition_ and errors_ attributes",
        hasattr(_dec, "decomposition_") and hasattr(_dec, "errors_"),
        diag=f"decomposition_ present={hasattr(_dec, 'decomposition_')}, errors_ present={hasattr(_dec, 'errors_')}"
    )
except Exception as exc:
    check("REQ-2b: fit_transform runs",       False, diag=str(exc))
    check("REQ-2c: attributes stored",        False, diag="(skipped)")

# ===========================================================================
# REQ-3  Stokes constraint: S0 >= ||(S1,S2,S3)|| and S0 >= 0
# REQ-4  Activation maps are non-negative in constrained model
# ===========================================================================

print("\n--- REQ-3 & REQ-4: Stokes and non-negativity constraints ---")

for _method in ("bpg", "ao_admm"):
    try:
        _T_stokes, _, _ = gen_ll1(
            (8, 6, 4), rank=2, stokes_constrained=True,
            non_negative_matrices=True, noise_level=1e-6, random_state=1234
        )
        _kw = dict(
            rank=2, n_iter_max=(200 if _method == "bpg" else 400),
            init="svd", stokes_vectors=True, non_negative_matrices=True,
            random_state=1234, return_errors=True,
        )
        if _method == "bpg":
            (_mC, _vC), _ = ll1_bpg(_T_stokes, **_kw)
        else:
            (_mC, _vC), _ = ll1_ao_admm(
                _T_stokes, n_iter_max_inner=20,
                tol_outer=1e-12, tol_inner=1e-10, **_kw
            )

        # REQ-3: Stokes validity
        _vC_np = tl.to_numpy(_vC)
        _ok, _bad_r, _s0, _spn = _stokes_valid(_vC_np)
        check(
            f"REQ-3 [{_method}]: output Stokes vectors satisfy S0>=||(S1,S2,S3)||",
            _ok,
            diag=(
                f"all valid" if _ok
                else f"component {_bad_r} violated: S0={_s0:.4f}, ||s_pol||={_spn:.4f}"
            )
        )

        # REQ-4: Non-negativity of A_r
        _nn_ok = all(
            float(tl.to_numpy(tl.min(m))) >= -1e-10 for m in _mC
        )
        _min_vals = [float(tl.to_numpy(tl.min(m))) for m in _mC]
        check(
            f"REQ-4 [{_method}]: activation matrices A_r are non-negative",
            _nn_ok,
            diag=f"min values: {[f'{v:.2e}' for v in _min_vals]}"
        )
    except Exception as exc:
        check(f"REQ-3 [{_method}]: Stokes validity",   False, diag=str(exc))
        check(f"REQ-4 [{_method}]: non-negativity",    False, diag="(skipped)")

# ===========================================================================
# REQ-5  check_ll1_uniqueness returns True / False
# ===========================================================================

print("\n--- REQ-5: check_ll1_uniqueness ---")

_cases = [
    # (shape,  rank, expected, description)
    ((10, 10, 5), 3,  True,  "K=5 >= R=3, IJ=100 >= R=3 → unique"),
    ((10, 10, 2), 5,  False, "K=2 < R=5  → not unique"),
    ((2,  2,  10), 10, False, "IJ=4 < R=10 → not unique"),
]
for _shape, _rank, _expected, _desc in _cases:
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            _got = check_ll1_uniqueness(_shape, _rank)
        check(
            f"REQ-5: uniqueness({_shape}, {_rank}) == {_expected}",
            _got is _expected,
            diag=f"expected={_expected}, got={_got} | {_desc}"
        )
    except Exception as exc:
        check(f"REQ-5: uniqueness({_shape}, {_rank})", False, diag=str(exc))

# ===========================================================================
# REQ-6  ll1_als  – exists and accepts the public interface
# ===========================================================================

print("\n--- REQ-6: ll1_als public interface ---")

try:
    _T6, _, _ = gen_ll1((6, 5, 4), rank=2, random_state=0)

    # return_errors=True
    _r6, _e6 = ll1_als(_T6, rank=2, n_iter_max=30, init="svd",
                       random_state=0, return_errors=True)
    check(
        "REQ-6a: ll1_als(return_errors=True) returns (result, list)",
        isinstance(_e6, list) and len(_e6) > 0,
        diag=f"len(errors)={len(_e6)}, last error={_e6[-1]:.4e}"
    )

    # Errors are non-increasing (ALS property)
    _monotone = all(_e6[i] <= _e6[i-1] + 1e-8 for i in range(1, len(_e6)))
    check(
        "REQ-6b: ll1_als errors are non-increasing",
        _monotone,
        diag=f"first={_e6[0]:.4e}, last={_e6[-1]:.4e}"
    )

    # init='random'
    _r6r = ll1_als(_T6, rank=2, n_iter_max=20, init="random", random_state=5)
    _m6r, _v6r = _r6r
    check(
        "REQ-6c: ll1_als accepts init='random'",
        len(_m6r) == 2,
        diag=f"OK – {len(_m6r)} matrices returned"
    )
except Exception as exc:
    check("REQ-6a: ll1_als return_errors", False, diag=str(exc))
    check("REQ-6b: errors non-increasing", False, diag="(skipped)")
    check("REQ-6c: init='random'",         False, diag="(skipped)")

# ===========================================================================
# REQ-7  ll1_bpg  – exists and accepts the public interface
# ===========================================================================

print("\n--- REQ-7: ll1_bpg public interface ---")

try:
    _T7, _, _ = gen_ll1((6, 5, 4), rank=2, stokes_constrained=True,
                        non_negative_matrices=True, random_state=0)
    _r7, _e7 = ll1_bpg(
        _T7, rank=2, n_iter_max=50, init="svd",
        stokes_vectors=True, non_negative_matrices=True,
        random_state=0, return_errors=True
    )
    _m7, _v7 = _r7
    check(
        "REQ-7a: ll1_bpg returns (matrices, vectors) and errors",
        len(_m7) == 2 and isinstance(_e7, list),
        diag=f"len(matrices)={len(_m7)}, len(errors)={len(_e7)}"
    )
    check(
        "REQ-7b: ll1_bpg accepts stokes_vectors and non_negative_matrices flags",
        True,   # if we got here without exception, flags were accepted
        diag="flags accepted"
    )
except Exception as exc:
    check("REQ-7a: ll1_bpg basic call", False, diag=str(exc))
    check("REQ-7b: ll1_bpg flags",      False, diag="(skipped)")

# ===========================================================================
# REQ-8  ll1_ao_admm  – exists and accepts the public interface
# ===========================================================================

print("\n--- REQ-8: ll1_ao_admm public interface ---")

try:
    _T8, _, _ = gen_ll1((6, 5, 4), rank=2, stokes_constrained=True,
                        non_negative_matrices=True, random_state=0)
    _r8, _e8 = ll1_ao_admm(
        _T8, rank=2, n_iter_max=30, n_iter_max_inner=5,
        init="svd", stokes_vectors=True, non_negative_matrices=True,
        random_state=0, return_errors=True
    )
    _m8, _v8 = _r8
    check(
        "REQ-8a: ll1_ao_admm returns (matrices, vectors) and errors",
        len(_m8) == 2 and isinstance(_e8, list),
        diag=f"len(matrices)={len(_m8)}, len(errors)={len(_e8)}"
    )
    check(
        "REQ-8b: ll1_ao_admm n_iter_max_inner parameter accepted",
        True,
        diag="parameter accepted without error"
    )
except Exception as exc:
    check("REQ-8a: ll1_ao_admm basic call",       False, diag=str(exc))
    check("REQ-8b: n_iter_max_inner accepted",     False, diag="(skipped)")

# ===========================================================================
# REQ-9  gen_ll1 in tensorly.datasets
# ===========================================================================

print("\n--- REQ-9: gen_ll1 in tensorly.datasets ---")

try:
    _T9u, _M9u, _V9u = gen_ll1((8, 6, 5), rank=3, random_state=7)
    check(
        "REQ-9a: gen_ll1 returns (tensor, matrices, vectors)",
        tl.shape(_T9u) == (8, 6, 5) and len(_M9u) == 3,
        diag=f"tensor shape={tl.shape(_T9u)}, R={len(_M9u)}"
    )

    # Noiseless tensor must match reconstruction exactly
    if _ll1_to_tensor is not None:
        _T9rec = _ll1_to_tensor(_M9u, _V9u)
        _err9 = _rel_error(_T9u, _T9rec)
        check(
            "REQ-9b: noiseless gen_ll1 tensor matches sum_r A_r⊗c_r exactly",
            _err9 < 1e-10,
            diag=f"rel_error={_err9:.2e}"
        )

    # Stokes-constrained generation
    _T9s, _M9s, _V9s = gen_ll1(
        (8, 6, 4), rank=2, stokes_constrained=True,
        non_negative_matrices=True, random_state=7
    )
    _V9s_np = tl.to_numpy(_V9s)
    _ok9, _bad9, _s09, _spn9 = _stokes_valid(_V9s_np)
    check(
        "REQ-9c: gen_ll1(stokes_constrained=True) yields valid Stokes vectors",
        _ok9,
        diag=(
            "all valid" if _ok9
            else f"component {_bad9}: S0={_s09:.4f}, ||s_pol||={_spn9:.4f}"
        )
    )
    _nn9 = all(float(tl.to_numpy(tl.min(m))) >= 0 for m in _M9s)
    check(
        "REQ-9d: gen_ll1(non_negative_matrices=True) yields non-negative A_r",
        _nn9,
        diag=f"min vals: {[float(tl.to_numpy(tl.min(m))) for m in _M9s]}"
    )

    # Noise injection
    _T9noisy, _, _ = gen_ll1(
        (6, 5, 4), rank=2, noise_level=0.1, random_state=7)
    _T9clean, _, _ = gen_ll1(
        (6, 5, 4), rank=2, noise_level=0.0, random_state=7)
    _diff9 = float(tl.to_numpy(tl.norm(_T9noisy - _T9clean)))
    check(
        "REQ-9e: gen_ll1 noise_level>0 produces a noisy tensor",
        _diff9 > 1e-6,
        diag=f"||noisy - clean||={_diff9:.4e}"
    )
except Exception as exc:
    check("REQ-9a: gen_ll1 shape",              False, diag=str(exc))
    check("REQ-9b: noiseless exact",            False, diag="(skipped)")
    check("REQ-9c: Stokes valid in gen_ll1",    False, diag="(skipped)")
    check("REQ-9d: non-negative A_r in gen_ll1", False, diag="(skipped)")
    check("REQ-9e: noise injection",            False, diag="(skipped)")

# ===========================================================================
# REQ-10  Unconstrained noiseless convergence: rel error < 1e-9
# ===========================================================================

print("\n--- REQ-10: ll1_als noiseless convergence < 1e-9 ---")

for _seed in (0, 1234):
    try:
        _T10, _, _ = gen_ll1((8, 6, 5), rank=2, random_state=_seed)
        (_m10, _v10), _ = ll1_als(
            _T10, rank=2, n_iter_max=2000, init="svd",
            tol=1e-13, random_state=_seed, return_errors=True
        )
        if _ll1_to_tensor is not None:
            _T10rec = _ll1_to_tensor(_m10, _v10)
            _e10 = _rel_error(_T10, _T10rec)
        else:
            _e10 = float(tl.to_numpy(
                tl.norm(_T10 - sum(
                    tl.reshape(_m10[r], (*tl.shape(_T10)[:2], 1)) *
                    tl.reshape(_v10[:, r], (1, 1, tl.shape(_T10)[2]))
                    for r in range(2)
                )) / tl.norm(_T10)
            ))
        check(
            f"REQ-10 [seed={_seed}]: unconstrained noiseless rel error < 1e-9",
            _e10 < 1e-9,
            diag=f"rel_error={_e10:.2e}"
        )
    except Exception as exc:
        check(f"REQ-10 [seed={_seed}]: convergence", False, diag=str(exc))

# ===========================================================================
# REQ-11  Constrained noisy convergence (noise_level=1e-6): rel error < 1e-4
# ===========================================================================

print("\n--- REQ-11: constrained noisy convergence < 1e-4 ---")

for _algo in ("bpg", "ao_admm"):
    try:
        _T11, _, _ = gen_ll1(
            (10, 8, 4), rank=2, stokes_constrained=True,
            non_negative_matrices=True, noise_level=1e-6, random_state=1234
        )
        _kw11 = dict(
            rank=2, init="svd",
            stokes_vectors=True, non_negative_matrices=True,
            random_state=1234, return_errors=True,
        )
        if _algo == "bpg":
            (_m11, _v11), _ = ll1_bpg(
                _T11, n_iter_max=500, tol=1e-10, **_kw11
            )
        else:
            (_m11, _v11), _ = ll1_ao_admm(
                _T11, n_iter_max=1000, n_iter_max_inner=20,
                tol_outer=1e-12, tol_inner=1e-10, **_kw11
            )

        if _ll1_to_tensor is not None:
            _T11rec = _ll1_to_tensor(_m11, _v11)
        else:
            _T11rec = sum(
                tl.reshape(_m11[r], (*tl.shape(_T11)[:2], 1)) *
                tl.reshape(_v11[:, r], (1, 1, tl.shape(_T11)[2]))
                for r in range(2)
            )
        _e11 = _rel_error(_T11, _T11rec)
        check(
            f"REQ-11 [{_algo}]: constrained noisy rel error < 1e-4",
            _e11 < 1e-4,
            diag=f"rel_error={_e11:.2e}"
        )
    except Exception as exc:
        check(f"REQ-11 [{_algo}]: convergence", False, diag=str(exc))

# ===========================================================================
# REQ-R  REGRESSION – existing CP/PARAFAC decomposition still works (Rule 5)
# ===========================================================================

print("\n--- REQ-R: regression – existing CP decomposition unaffected ---")

if _cp_available:
    try:
        _rng_r = tl.check_random_state(42)
        _T_cp = tl.tensor(_rng_r.random_sample((6, 5, 4)))
        _cp = parafac(_T_cp, rank=3, n_iter_max=50, random_state=42)
        _T_cp_rec = cp_to_tensor(_cp)
        _e_cp = _rel_error(_T_cp, _T_cp_rec)
        check(
            "REQ-R: parafac still runs and returns a valid reconstruction",
            _e_cp < 0.5,   # very loose – just ensure it didn't break
            diag=f"rel_error={_e_cp:.4f}"
        )
        check(
            "REQ-R: CP reconstruction has correct shape",
            tl.shape(_T_cp_rec) == (6, 5, 4),
            diag=f"shape={tl.shape(_T_cp_rec)}"
        )
    except Exception as exc:
        check("REQ-R: parafac runs",     False, diag=str(exc))
        check("REQ-R: CP shape correct", False, diag="(skipped)")
else:
    check("REQ-R: parafac available", False,
          diag="parafac could not be imported")

# ===========================================================================
# Write reward and exit (Rule 8)
# ===========================================================================

write_reward_and_exit()

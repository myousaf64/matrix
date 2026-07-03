# matrix ("Enter the Matrix") — progress

**Status: all 14 mandatory exercises (ex00–ex13) DONE & verified.**

`matrix.py` — `Vector` and `Matrix` over the reals:
- ex00 add/sub/scl · ex01 linear_combination · ex02 lerp (scalar/vector/matrix) ·
  ex03 dot · ex04 norm_1/norm/norm_inf · ex05 angle_cos · ex06 cross_product
- ex07 mul_vec/mul_mat · ex08 trace · ex09 transpose · ex10 row_echelon (RREF) ·
  ex11 determinant (≤4×4) · ex12 inverse · ex13 rank

Complexity bounds respected: dot/norm/trace O(n), matmul O(nmp), echelon/det/inverse/
rank via Gaussian elimination O(n³). Only elementary ops + `pow` (allowed for norm).

`test_matrix.py` — asserts every subject worked example (tolerance 1e-4). It's also
the runnable demo. **Run:** `python3 test_matrix.py`

## Next (bonus, only if mandatory stays perfect)
- ex14: projection matrix (needs `tan`)
- ex15: redo generically with K = your own Complex type (design is already
  operator-based, so mostly swapping float ops for a Complex class)

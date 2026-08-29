# matrix

Linear algebra from scratch: `Vector` and `Matrix` types over the reals, covering
all fourteen mandatory exercises of the 42 matrix project. A 42 Abu Dhabi project.

## Run

```
python3 test_matrix.py
```

The test file asserts every worked example from the assignment and doubles as the
runnable demo.

## Contents

Addition, subtraction and scaling; linear combination; linear interpolation; dot
product; the 1, 2 and infinity norms; cosine of the angle; cross product; matrix
and vector multiplication; trace; transpose; reduced row echelon form;
determinant up to 4x4; inverse; rank.

## Notes

- Complexity bounds from the subject are respected: O(n) for dot, norm and trace,
  O(nmp) for multiplication, O(n^3) for elimination, determinant, inverse and rank.
- Elementary operations only, plus `pow`, which the subject allows for norms.
- `PROGRESS.md` is the development log.

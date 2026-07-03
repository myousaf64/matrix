#!/usr/bin/env python3
"""Enter the Matrix — vectors & matrices over the reals (f32), from scratch.

Covers ex00-ex13. Only elementary ops plus pow/max are used (pow is allowed for
the norm exercise); no linear-algebra library does the work. Operations meet the
subject's complexity bounds: dot/norm/trace are O(n), matmul O(nmp), and
row-echelon / determinant / inverse / rank use Gaussian elimination in O(n^3).
"""
EPS = 1e-10


class Vector:
    def __init__(self, data):
        self.data = [float(x) for x in data]

    def size(self):
        return len(self.data)

    # --- ex00: in-place add / sub / scale ---
    def add(self, v):
        for i in range(self.size()):
            self.data[i] += v.data[i]
        return self

    def sub(self, v):
        for i in range(self.size()):
            self.data[i] -= v.data[i]
        return self

    def scl(self, a):
        for i in range(self.size()):
            self.data[i] *= a
        return self

    # operator forms return new vectors (used by lerp)
    def __add__(self, v):
        return Vector([a + b for a, b in zip(self.data, v.data)])

    def __sub__(self, v):
        return Vector([a - b for a, b in zip(self.data, v.data)])

    def __mul__(self, s):
        return Vector([a * s for a in self.data])

    # --- ex03: dot product (O(n)) ---
    def dot(self, v):
        return sum(a * b for a, b in zip(self.data, v.data))

    # --- ex04: norms (O(n)) ---
    def norm_1(self):
        return sum(abs(a) for a in self.data)

    def norm(self):
        return sum(a * a for a in self.data) ** 0.5

    def norm_inf(self):
        return max((abs(a) for a in self.data), default=0.0)

    def __str__(self):
        return '\n'.join(f'[{a:g}]' for a in self.data)


class Matrix:
    def __init__(self, rows):
        self.rows = [[float(x) for x in r] for r in rows]

    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def is_square(self):
        m, n = self.shape()
        return m == n

    def _copy(self):
        return [r[:] for r in self.rows]

    # --- ex00: in-place add / sub / scale ---
    def add(self, m):
        for i, row in enumerate(self.rows):
            for j in range(len(row)):
                row[j] += m.rows[i][j]
        return self

    def sub(self, m):
        for i, row in enumerate(self.rows):
            for j in range(len(row)):
                row[j] -= m.rows[i][j]
        return self

    def scl(self, a):
        for row in self.rows:
            for j in range(len(row)):
                row[j] *= a
        return self

    def __add__(self, m):
        return Matrix([[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self.rows, m.rows)])

    def __sub__(self, m):
        return Matrix([[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(self.rows, m.rows)])

    def __mul__(self, s):
        return Matrix([[a * s for a in r] for r in self.rows])

    # --- ex07: matrix * vector (O(nm)) and matrix * matrix (O(nmp)) ---
    def mul_vec(self, vec):
        return Vector([sum(a * b for a, b in zip(row, vec.data)) for row in self.rows])

    def mul_mat(self, mat):
        m, n = self.shape()
        _, p = mat.shape()
        out = [[0.0] * p for _ in range(m)]
        for i in range(m):
            for k in range(n):
                aik = self.rows[i][k]
                for j in range(p):
                    out[i][j] += aik * mat.rows[k][j]
        return Matrix(out)

    # --- ex08: trace (O(n)) ---
    def trace(self):
        return sum(self.rows[i][i] for i in range(len(self.rows)))

    # --- ex09: transpose (O(nm)) ---
    def transpose(self):
        m, n = self.shape()
        return Matrix([[self.rows[i][j] for i in range(m)] for j in range(n)])

    # --- ex10: reduced row-echelon form (O(n^3)) ---
    def row_echelon(self):
        a = self._copy()
        rows = len(a)
        cols = len(a[0]) if rows else 0
        lead = 0
        for r in range(rows):
            if lead >= cols:
                break
            i = r
            while abs(a[i][lead]) < EPS:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        return Matrix(a)
            a[i], a[r] = a[r], a[i]
            piv = a[r][lead]
            a[r] = [x / piv for x in a[r]]
            for i in range(rows):
                if i != r:
                    f = a[i][lead]
                    a[i] = [a[i][c] - f * a[r][c] for c in range(cols)]
            lead += 1
        return Matrix(a)

    # --- ex11: determinant, dim <= 4 (O(n^3) elimination with sign) ---
    def determinant(self):
        a = self._copy()
        n = len(a)
        det = 1.0
        for i in range(n):
            p = max(range(i, n), key=lambda r: abs(a[r][i]))
            if abs(a[p][i]) < EPS:
                return 0.0
            if p != i:
                a[i], a[p] = a[p], a[i]
                det = -det
            det *= a[i][i]
            for r in range(i + 1, n):
                f = a[r][i] / a[i][i]
                for c in range(i, n):
                    a[r][c] -= f * a[i][c]
        return det

    # --- ex12: inverse via Gauss-Jordan on [A|I] (O(n^3)) ---
    def inverse(self):
        n = len(self.rows)
        aug = [self.rows[i][:] + [1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]
        for i in range(n):
            p = max(range(i, n), key=lambda r: abs(aug[r][i]))
            if abs(aug[p][i]) < EPS:
                raise ValueError('matrix is singular, no inverse')
            aug[i], aug[p] = aug[p], aug[i]
            piv = aug[i][i]
            aug[i] = [x / piv for x in aug[i]]
            for r in range(n):
                if r != i:
                    f = aug[r][i]
                    aug[r] = [aug[r][c] - f * aug[i][c] for c in range(2 * n)]
        return Matrix([row[n:] for row in aug])

    # --- ex13: rank = number of nonzero rows in row-echelon form (O(n^3)) ---
    def rank(self):
        ref = self.row_echelon().rows
        return sum(1 for row in ref if any(abs(x) > EPS for x in row))

    def __str__(self):
        return '\n'.join('[' + ', '.join(f'{x:g}' for x in r) + ']' for r in self.rows)


# --- ex01: linear combination (O(n)) ---
def linear_combination(vectors, coefs):
    dim = vectors[0].size()
    out = [0.0] * dim
    for vec, c in zip(vectors, coefs):
        for i in range(dim):
            out[i] += vec.data[i] * c   # fused multiply-add
    return Vector(out)


# --- ex02: linear interpolation, generic over scalars / vectors / matrices ---
def lerp(u, v, t):
    return u + (v - u) * t


# --- ex05: cosine of the angle between two vectors ---
def angle_cos(u, v):
    return u.dot(v) / (u.norm() * v.norm())


# --- ex06: cross product of two 3D vectors ---
def cross_product(u, v):
    a, b = u.data, v.data
    return Vector([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])

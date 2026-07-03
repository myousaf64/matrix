"""Self-check against every subject worked example. Run: python3 test_matrix.py"""
from matrix import (Vector, Matrix, linear_combination, lerp, angle_cos,
                    cross_product)

T = 1e-4


def close(a, b):
    return abs(a - b) < T


def vclose(v, expected):
    return len(v.data) == len(expected) and all(close(a, b) for a, b in zip(v.data, expected))


def mclose(m, expected):
    return all(all(close(a, b) for a, b in zip(r1, r2)) for r1, r2 in zip(m.rows, expected))


def main():
    # ex00 vectors
    u = Vector([2., 3.]); u.add(Vector([5., 7.])); assert vclose(u, [7., 10.])
    u = Vector([2., 3.]); u.sub(Vector([5., 7.])); assert vclose(u, [-3., -4.])
    u = Vector([2., 3.]); u.scl(2.); assert vclose(u, [4., 6.])
    # ex00 matrices
    a = Matrix([[1., 2.], [3., 4.]]); a.add(Matrix([[7., 4.], [-2., 2.]]))
    assert mclose(a, [[8., 6.], [1., 6.]])
    a = Matrix([[1., 2.], [3., 4.]]); a.scl(2.); assert mclose(a, [[2., 4.], [6., 8.]])

    # ex01 linear combination
    e1, e2, e3 = Vector([1., 0., 0.]), Vector([0., 1., 0.]), Vector([0., 0., 1.])
    assert vclose(linear_combination([e1, e2, e3], [10., -2., 0.5]), [10., -2., 0.5])
    v1, v2 = Vector([1., 2., 3.]), Vector([0., 10., -100.])
    assert vclose(linear_combination([v1, v2], [10., -2.]), [10., 0., 230.])

    # ex02 lerp (scalars, vectors, matrices)
    assert close(lerp(0., 1., 0.5), 0.5)
    assert close(lerp(21., 42., 0.3), 27.3)
    assert vclose(lerp(Vector([2., 1.]), Vector([4., 2.]), 0.3), [2.6, 1.3])
    assert mclose(lerp(Matrix([[2., 1.], [3., 4.]]), Matrix([[20., 10.], [30., 40.]]), 0.5),
                  [[11., 5.5], [16.5, 22.]])

    # ex03 dot
    assert close(Vector([-1., 6.]).dot(Vector([3., 2.])), 9.)
    # ex04 norms
    w = Vector([1., 2., 3.])
    assert close(w.norm_1(), 6.) and close(w.norm(), 3.7416574) and close(w.norm_inf(), 3.)
    # ex05 cosine
    assert close(angle_cos(Vector([-1., 1.]), Vector([1., -1.])), -1.)
    # ex06 cross product
    assert vclose(cross_product(Vector([1., 2., 3.]), Vector([4., 5., 6.])), [-3., 6., -3.])

    # ex07 matmul
    assert vclose(Matrix([[2., -2.], [-2., 2.]]).mul_vec(Vector([4., 2.])), [4., -4.])
    assert mclose(Matrix([[3., -5.], [6., 8.]]).mul_mat(Matrix([[2., 1.], [4., 2.]])),
                  [[-14., -7.], [44., 22.]])
    # ex08 trace
    assert close(Matrix([[-2., -8., 4.], [1., -23., 4.], [0., 6., 4.]]).trace(), -21.)
    # ex09 transpose
    assert mclose(Matrix([[1., 2., 3.], [4., 5., 6.]]).transpose(),
                  [[1., 4.], [2., 5.], [3., 6.]])
    # ex10 row echelon
    assert mclose(Matrix([[1., 2.], [3., 4.]]).row_echelon(), [[1., 0.], [0., 1.]])
    assert mclose(Matrix([[8., 5., -2., 4., 28.], [4., 2.5, 20., 4., -4.], [8., 5., 1., 4., 17.]]).row_echelon(),
                  [[1., 0.625, 0., 0., -12.1666667], [0., 0., 1., 0., -3.6666667], [0., 0., 0., 1., 29.5]])
    # ex11 determinant
    assert close(Matrix([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]).determinant(), 8.)
    assert close(Matrix([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]]).determinant(), -174.)
    assert close(Matrix([[8., 5., -2., 4.], [4., 2.5, 20., 4.], [8., 5., 1., 4.], [28., -4., 17., 1.]]).determinant(), 1032.)
    # ex12 inverse
    assert mclose(Matrix([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]]).inverse(),
                  [[0.649425287, 0.097701149, -0.655172414],
                   [-0.781609195, -0.126436782, 0.965517241],
                   [0.143678161, 0.074712644, -0.206896552]])
    # ex13 rank
    assert Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).rank() == 3
    assert Matrix([[1., 2., 0., 0.], [2., 4., 0., 0.], [-1., 2., 1., 1.]]).rank() == 2
    assert Matrix([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.], [21., 18., 7.]]).rank() == 3

    print('OK: all matrix exercises (ex00-ex13) pass')


if __name__ == '__main__':
    main()

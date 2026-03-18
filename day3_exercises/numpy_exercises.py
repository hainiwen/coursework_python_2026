# Exercise 2: NumPy Exercises


import numpy as np

def section(label):
    print(f"\n{'=' * 50}")
    print(f" {label}")
    print('=' * 50)


# -----------------------------------------------------------------------
# 2a. Null vector of size 10, fifth value = 1
# -----------------------------------------------------------------------
section("2a: Null vector, fifth value = 1")
Z = np.zeros(10)
Z[4] = 1
print(Z)
# Expected: [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]


# -----------------------------------------------------------------------
# 2b. Vector with values ranging from 10 to 49
# -----------------------------------------------------------------------
section("2b: Values 10 to 49")
Z = np.arange(10, 50)
print(Z)
# Expected: [10 11 12 ... 49]


# -----------------------------------------------------------------------
# 2c. Reverse a vector
# -----------------------------------------------------------------------
section("2c: Reverse a vector")
Z = np.arange(10)
Z = Z[::-1]
print(Z)
# Expected: [9 8 7 6 5 4 3 2 1 0]


# -----------------------------------------------------------------------
# 2d. 3x3 matrix with values 0 to 8
# -----------------------------------------------------------------------
section("2d: 3x3 matrix, values 0-8")
Z = np.arange(9).reshape(3, 3)
print(Z)
# Expected:
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]


# -----------------------------------------------------------------------
# 2e. Find indices of non-zero elements from [1,2,0,0,4,0]
# -----------------------------------------------------------------------
section("2e: Non-zero indices of [1,2,0,0,4,0]")
result = np.nonzero([1, 2, 0, 0, 4, 0])
print(result)
# Expected: (array([0, 1, 4]),)


# -----------------------------------------------------------------------
# 2f. Random vector of size 30, find mean value
# -----------------------------------------------------------------------
section("2f: Random vector size 30, mean")
np.random.seed(42)          
Z = np.random.random(30)
print(Z.mean())
# Expected: some float between 0 and 1


# -----------------------------------------------------------------------
# 2g. 2D array with 1 on the border and 0 inside
# -----------------------------------------------------------------------
section("2g: 2D array, 1 on border, 0 inside")
Z = np.ones((6, 6))
Z[1:-1, 1:-1] = 0           # set interior to 0
print(Z)
# Expected: 6x6 array with 1s on the edges and 0s in the 4x4 interior


# -----------------------------------------------------------------------
# 2h. 8x8 checkerboard pattern (manual indexing)
# -----------------------------------------------------------------------
section("2h: 8x8 checkerboard (manual)")
Z = np.zeros((8, 8), dtype=int)
Z[1::2, ::2] = 1            # odd rows, even cols
Z[::2, 1::2] = 1            # even rows, odd cols
print(Z)


# -----------------------------------------------------------------------
# 2i. 8x8 checkerboard using np.tile
# -----------------------------------------------------------------------
section("2i: 8x8 checkerboard using tile")
Z = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
print(Z)
# Should match 2h exactly — verify by rebuilding the manual version
ref = np.zeros((8, 8), dtype=int)
ref[1::2, ::2] = 1
ref[::2, 1::2] = 1
assert np.array_equal(Z, ref)


# -----------------------------------------------------------------------
# 2j. Negate all elements between 3 and 8, in place
# -----------------------------------------------------------------------
section("2j: Negate elements strictly between 3 and 8")
Z = np.arange(11)
print("Before:", Z)
Z[(Z > 3) & (Z < 8)] *= -1  # in-place, no copy created
print("After: ", Z)


# -----------------------------------------------------------------------
# 2k. Create a random vector of size 10 and sort it
# -----------------------------------------------------------------------
section("2k: Sort random vector of size 10")
np.random.seed(0)
Z = np.random.random(10)
Z = np.sort(Z)
print(Z)


# -----------------------------------------------------------------------
# 2l. Check if two random arrays are equal
# -----------------------------------------------------------------------
section("2l: Check if two arrays are equal")
np.random.seed(1)
A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
equal = np.array_equal(A, B)
print(f"A:     {A}")
print(f"B:     {B}")
print(f"Equal: {equal}")


# -----------------------------------------------------------------------
# 2m. Square every element in place (no temporaries, dtype preserved)
# -----------------------------------------------------------------------
section("2m: Square in place, no temporaries")
Z = np.arange(10, dtype=np.int32)
print(f"Before: {Z}  dtype={Z.dtype}")
np.multiply(Z, Z, out=Z)    # writes result back into Z, no temp array
print(f"After:  {Z}  dtype={Z.dtype}")



# -----------------------------------------------------------------------
# 2n. Get the diagonal of a dot product (efficiently)
# -----------------------------------------------------------------------
section("2n: Diagonal of a dot product")
A = np.arange(9).reshape(3, 3)
B = A + 1
C = np.dot(A, B)            # full matrix product (for reference)

# Efficient: einsum computes only the diagonal elements
D = np.einsum('ij,ji->i', A, B)

print(f"A:\n{A}")
print(f"B:\n{B}")
print(f"Full C (np.dot):\n{C}")
print(f"Diagonal via einsum: {D}")
print(f"Verify np.diag(C):   {np.diag(C)}")
assert np.array_equal(D, np.diag(C))
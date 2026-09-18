
import math

MAX_ITER = 1000
EPS = 1e-8
# Matrix-vector multiplication
def mat_vec_mul(A, x):
    n = len(A)
    y = [0] * n
    for i in range(n):
        for j in range(n):
            y[i] += A[i][j] * x[j]
    return y

# Normalize vector
def normalize(v):
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        return v
    return [x / norm for x in v]

# Power Iteration
def power_iteration(A):
    n = len(A)
    x = [1.0] * n

    for _ in range(MAX_ITER):
        y = mat_vec_mul(A, x)
        y = normalize(y)

        error = sum(abs(y[i] - x[i]) for i in range(n))
        x = y

        if error < EPS:
            break

    # Rayleigh Quotient (Eigenvalue)
    Ax = mat_vec_mul(A, x)

    numerator = sum(x[i] * Ax[i] for i in range(n))
    denominator = sum(x[i] * x[i] for i in range(n))

    eigenvalue = numerator / denominator

    return eigenvalue, x

# Deflation
def deflate(A, eigenvalue, eigenvector):
    n = len(A)

    for i in range(n):
        for j in range(n):
            A[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]

# Main Program
n = int(input("Enter size of matrix (N): "))

A = []
print("Enter matrix elements row-wise:")

for i in range(n):
    row = list(map(float, input().split()))
    A.append(row)

k = int(input("Enter value of K: "))

eigenvalues = []
eigenvectors = []

for _ in range(k):
    val, vec = power_iteration(A)
    eigenvalues.append(val)
    eigenvectors.append(vec)
    deflate(A, val, vec)

print("\nEigenvalues:")
for i in range(k):
    print(f"Eigenvalue {i+1}: {eigenvalues[i]:.6f}")

print("\nEigenvectors:")
for i in range(k):
    print(f"Eigenvector {i+1}:")
    for x in eigenvectors[i]:
        print(f"{x:.6f}", end=" ")
    print()

print("\nTop", k, "Principal Components:")
for i in range(k):
    print(f"PC {i+1}:")
    for x in eigenvectors[i]:
        print(f"{x:.6f}", end=" ")
    print()
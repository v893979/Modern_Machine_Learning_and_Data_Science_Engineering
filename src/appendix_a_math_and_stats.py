"""
Appendix A: Mathematics & Statistics Primer for Data Scientists
Modern Machine Learning & Data Science Engineering
"""
import numpy as np

def demonstrate_linear_algebra():
    print("=" * 65)
    print("1. MATRIX MULTIPLICATION & DETERMINANTS")
    print("=" * 65)
    A = np.array([[2, 3], [1, 4]])
    B = np.array([[5, 1], [2, 6]])
    C = np.dot(A, B) # Matrix product
    det_A = np.linalg.det(A)
    inv_A = np.linalg.inv(A)

    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("Matrix Product A @ B:\n", C)
    print(f"Determinant of A: {det_A:.2f}")
    print("Inverse of A:\n", inv_A)

    # Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("\nEigenvalues of A:", np.round(eigenvalues, 4))
    print("Eigenvectors of A:\n", np.round(eigenvectors, 4))

def demonstrate_symbolic_calculus():
    print("\n" + "=" * 65)
    print("2. SYMBOLIC CALCULUS & GRADIENTS (SYMPY)")
    print("=" * 65)
    try:
        import sympy as sp
        x, y = sp.symbols('x y')
        f = x**3 - 5*x**2 + 6*x + 10
        df_dx = sp.diff(f, x)
        crit_points = sp.solve(df_dx, x)
        print(f"Function:              f(x) = {f}")
        print(f"First Derivative:     f'(x) = {df_dx}")
        print(f"Critical Points (f'=0): x = {crit_points}")

        # Multivariate Partial Derivatives
        g = 3*x**2 * y + sp.sin(x*y)
        dg_dx = sp.diff(g, x)
        dg_dy = sp.diff(g, y)
        print(f"\nMultivariate Function: g(x, y) = {g}")
        print(f"Partial ∂g/∂x: {dg_dx}")
        print(f"Partial ∂g/∂y: {dg_dy}")
    except ImportError:
        print("SymPy not installed. Run: pip install sympy")

def simulate_monty_hall(num_trials: int = 100000):
    print("\n" + "=" * 65)
    print(f"3. MONTE CARLO SIMULATION: MONTY HALL PROBLEM ({num_trials:,} TRIALS)")
    print("=" * 65)
    winning_door = np.random.randint(0, 3, size=num_trials)
    initial_pick = np.random.randint(0, 3, size=num_trials)

    # If staying with initial pick, wins whenever initial_pick == winning_door
    stay_wins = np.mean(initial_pick == winning_door)

    # If switching, always wins whenever initial_pick != winning_door
    switch_wins = np.mean(initial_pick != winning_door)

    print(f"Win Probability if STAYING:   {stay_wins * 100:.2f}% (Theoretical: 33.33%)")
    print(f"Win Probability if SWITCHING: {switch_wins * 100:.2f}% (Theoretical: 66.67%)")
    print("Conclusion: Switching doors literally doubles your winning odds!")

if __name__ == "__main__":
    demonstrate_linear_algebra()
    demonstrate_symbolic_calculus()
    simulate_monty_hall()

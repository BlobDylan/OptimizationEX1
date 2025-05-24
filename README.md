# Numerical Optimization with Python – EX1

This project is the first programming assignment for the **Numerical Optimization with Python (2025B)** course. The goal is to implement and test line search minimization methods, specifically Gradient Descent and Newton’s Method, with backtracking line search using Wolfe conditions.

## Features

- Modular implementation of unconstrained optimization methods.
- Includes:
  - Gradient Descent
  - Newton’s Method
- Supports:
  - Termination criteria based on parameter and objective function tolerances
  - Visualization of objective contours and iteration paths
  - Comparison of function value reduction across methods
- Test coverage using Python's `unittest` framework

## How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/BlobDylan/OptimizationEX1
   cd <repo-directory>
   ```

2. **Install dependencies**

   ```bash
    pip install -r requirements.txt
   ```

3. **Run tests**

   ```bash
    python -m unittest discover tests/

   ```

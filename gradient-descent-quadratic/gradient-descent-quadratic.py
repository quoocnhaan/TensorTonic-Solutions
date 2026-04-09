def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    #df_dx = 2ax + b
    for step in range(steps):
        df_dx = 2*x0*a + b
        x0 = x0 - lr * df_dx
    return x0
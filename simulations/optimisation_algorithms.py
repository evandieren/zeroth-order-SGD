import numpy as np
import sys
import time

def logistic_f(w, X, Y):
    output = 0
    for k in range(len(X)):
        output += np.log(1 + np.exp(- (w.T @ X[k]) * Y[k]))
    output = output / len(X)
    return output

def logistic_df(w, x, y):
    grad = - (1 / (1 + np.exp((w.T @ x) * y))) * y * x
    return grad

def GDClassifer(f, df, X, Y, w_0, eta, stopping_times, random_state):
    # Set Random State
    _ = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    # Unpack stopping times
    max_iter, max_grads = stopping_times

    # Setup
    time_ls = [0]
    loss_ls = [f(w_0, X, Y)]
    w_ls = [w_0]
    grad_count_ls = [0]

    w_i = w_0
    start_time = time.time()
    if max_grads is None:
        # For iterations
        for i in range(max_iter):
            # Average gradient over dataset
            grad = 0
            for k in range(len(X)):
                grad += logistic_df(w_i, X[k], Y[k])
            grad = grad / len(X)

            # Update
            w_i = w_i - eta * grad

            # Catalogue
            time_ls.append(time.time() - start_time)
            loss_ls.append(logistic_f(w_i, X, Y))
            w_ls.append(w_i)
            grad_count_ls.append(grad_count_ls[-1] + len(X))
    else:
        while grad_count_ls[-1] <= max_grads:
            # Average gradient over dataset
            grad = 0
            for k in range(len(X)):
                grad += logistic_df(w_i, X[k], Y[k])
            grad = grad / len(X)

            # Update
            w_i = w_i - eta * grad

            # Catalogue
            time_ls.append(time.time() - start_time)
            loss_ls.append(logistic_f(w_i, X, Y))
            w_ls.append(w_i)
            grad_count_ls.append(grad_count_ls[-1] + len(X))
    
    return time_ls, loss_ls, w_ls, grad_count_ls

def SGDClassifer(f, df, X, Y, w_0, eta, stopping_times, random_state):
    # Set Random State
    _ = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    # Unpack stopping times
    max_iter, max_grads = stopping_times

    # Setup
    time_ls = [0]
    loss_ls = [f(w_0, X, Y)]
    w_ls = [w_0]
    grad_count_ls = [0]

    w_i = w_0
    start_time = time.time()
    if max_grads is None:
        # For iterations
        for i in range(max_iter):
            k = np.random.randint(0, len(X))
            grad = logistic_df(w_i, X[k], Y[k])
            w_i = w_i - eta * logistic_df(w_i, X[k], Y[k])

            # Catalogue
            time_ls.append(time.time() - start_time)
            loss_ls.append(logistic_f(w_i, X, Y))
            w_ls.append(w_i)
            grad_count_ls.append(grad_count_ls[-1] + 1)
    else:
        while grad_count_ls[-1] <= max_grads:
            k = np.random.randint(0, len(X))
            grad = logistic_df(w_i, X[k], Y[k])
            w_i = w_i - eta * logistic_df(w_i, X[k], Y[k])

            # Catalogue
            time_ls.append(time.time() - start_time)
            loss_ls.append(logistic_f(w_i, X, Y))
            w_ls.append(w_i)
            grad_count_ls.append(grad_count_ls[-1] + 1)
    
    return time_ls, loss_ls, w_ls, grad_count_ls

def SVRGClassifer(f, df, X, Y, w_0, eta, m, stopping_times, random_state):
    # Set Random State
    _ = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    # Unpack stopping times
    max_iter, max_grads = stopping_times

    # Setup
    time_ls = [0]
    loss_ls = [f(w_0, X, Y)]
    w_ls = [w_0]
    grad_count_ls = [0]

    w_i = w_0
    start_time = time.time()
    if max_grads is None:
        # For iterations
        for i in range(max_iter):
            # Calc and store total gradient
            total_grad_dict = {}
            total_grad = 0
            for k in range(len(X)):
                temp_grad = logistic_df(w_i, X[k], Y[k])
                total_grad_dict[k] = temp_grad
                total_grad += temp_grad
            total_grad = total_grad / len(X)

            # Inner Loop
            dw_j = w_i
            for j in range(m):
                k = np.random.randint(0, len(X))
                grad = logistic_df(dw_j, X[k], Y[k])
                dw_j = dw_j - eta * (total_grad + grad - total_grad_dict[k])
            w_i = dw_j

            # Catalogue
            time_ls.append(time.time() - start_time)
            loss_ls.append(logistic_f(w_i, X, Y))
            w_ls.append(w_i)
            grad_count_ls.append(grad_count_ls[-1] + len(X) + m)
    else:
        while grad_count_ls[-1] <= max_grads:
            # Calc and store total gradient
            total_grad_dict = {}
            total_grad = 0
            for k in range(len(X)):
                temp_grad = logistic_df(w_i, X[k], Y[k])
                total_grad_dict[k] = temp_grad
                total_grad += temp_grad
            total_grad = total_grad / len(X)

            # Catalogue
            time_ls.append(time.time() - start_time)
            loss_ls.append(logistic_f(w_i, X, Y))
            w_ls.append(w_i)
            grad_count_ls.append(grad_count_ls[-1] + len(X))

            # Inner Loop
            dw_j = w_i
            for j in range(m):
                k = np.random.randint(0, len(X))
                grad = logistic_df(dw_j, X[k], Y[k])
                dw_j = dw_j - eta * (total_grad + grad - total_grad_dict[k])

                # Catalogue
                time_ls.append(time.time() - start_time)
                loss_ls.append(logistic_f(dw_j, X, Y))
                w_ls.append(dw_j)
                grad_count_ls.append(grad_count_ls[-1] + 1)
            
            # Return to Outer Loop
            w_i = dw_j
    
    return time_ls, loss_ls, w_ls, grad_count_ls

def SVRGClassifer_Alt(f, df, X, Y, w_0, eta, m, stopping_times, random_state):
    # Set Random State
    _ = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    # Unpack stopping times
    max_iter, max_grads = stopping_times

    # Setup
    time_ls = [0]
    loss_ls = [f(w_0, X, Y)]
    w_ls = [w_0]
    grad_count_ls = [0]

    w_i = w_0
    start_time = time.time()
    if max_grads is None:
        # For iterations
        for i in range(max_iter):
            # Check if i divisible by m
            if i % m == 0:
                # Calc and store total gradient
                total_grad_dict = {}
                total_grad = 0
                for k in range(len(X)):
                    temp_grad = logistic_df(w_i, X[k], Y[k])
                    total_grad_dict[k] = temp_grad
                    total_grad += temp_grad
                total_grad = total_grad / len(X)
            
            # Otherwise inner loop
            k = np.random.randint(0, len(X))
            grad = logistic_df(w_i, X[k], Y[k])
            w_i = w_i - eta * (total_grad + grad - total_grad_dict[k])

            # Catalogue
            time_ls.append(time.time() - start_time)
            loss_ls.append(logistic_f(w_i, X, Y))
            w_ls.append(w_i)
            if i % m == 0: 
                grad_count_ls.append(grad_count_ls[-1] + len(X) + 1)
            else:
                grad_count_ls.append(grad_count_ls[-1] + 1)
    else:
        i = 0
        while grad_count_ls[-1] <= max_grads:
            # Check if i divisible by m
            if i % m == 0:
                # Calc and store total gradient
                total_grad_dict = {}
                total_grad = 0
                for k in range(len(X)):
                    temp_grad = logistic_df(w_i, X[k], Y[k])
                    total_grad_dict[k] = temp_grad
                    total_grad += temp_grad
                total_grad = total_grad / len(X)
            
            # Otherwise inner loop
            k = np.random.randint(0, len(X))
            grad = logistic_df(w_i, X[k], Y[k])
            w_i = w_i - eta * (total_grad + grad - total_grad_dict[k])

            # Catalogue
            time_ls.append(time.time() - start_time)
            loss_ls.append(logistic_f(w_i, X, Y))
            w_ls.append(w_i)
            if i % m == 0: 
                grad_count_ls.append(grad_count_ls[-1] + len(X) + 1)
            else:
                grad_count_ls.append(grad_count_ls[-1] + 1)
            # Increment counter
            i += 1
    
    return time_ls, loss_ls, w_ls, grad_count_ls

def SAGAClassifer(f, df, X, Y, w_0, eta, stopping_times, random_state):
    # Set Random State
    _ = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    # Unpack stopping times
    max_iter, max_grads = stopping_times

    # Setup
    time_ls = [0]
    loss_ls = [f(w_0, X, Y)]
    w_ls = [w_0]
    grad_count_ls = [len(X)]

    w_i = w_0
    start_time = time.time()

    # Store total grad
    total_grad = 0
    grad_dict = {}
    for k in range(len(X)):
        grad_dict[k] = logistic_df(w_i, X[k], Y[k])
        total_grad += grad_dict[k]
    total_grad = total_grad / len(X)

    if max_grads is None:
        # For iterations
        for i in range(max_iter):
            k = np.random.randint(0, len(X))
            grad = logistic_df(w_i, X[k], Y[k])
            w_i = w_i - eta * (grad - grad_dict[k] + total_grad)

            # Update total_grad and grad_dict
            total_grad = ((len(X) * total_grad) - grad_dict[k] + grad) / len(X)
            grad_dict[k] = grad

            # Catalogue
            time_ls.append(time.time() - start_time)
            loss_ls.append(logistic_f(w_i, X, Y))
            w_ls.append(w_i)
            grad_count_ls.append(grad_count_ls[-1] + 1)
    else:
        while grad_count_ls[-1] <= max_grads:
            k = np.random.randint(0, len(X))
            grad = logistic_df(w_i, X[k], Y[k])
            w_i = w_i - eta * (grad - grad_dict[k] + total_grad)

            # Update total_grad and grad_dict
            total_grad = ((len(X) * total_grad) - grad_dict[k] + grad) / len(X)
            grad_dict[k] = grad

            # Catalogue
            time_ls.append(time.time() - start_time)
            loss_ls.append(logistic_f(w_i, X, Y))
            w_ls.append(w_i)
            grad_count_ls.append(grad_count_ls[-1] + 1)
    
    return time_ls, loss_ls, w_ls, grad_count_ls
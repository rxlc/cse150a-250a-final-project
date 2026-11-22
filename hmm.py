import numpy as np

def emission(B, x):
    """b_i(x) for every state i, returns all ones if x is -1 (missing observation)"""
    if x < 0:
        return np.ones(B.shape[0])
    return B[:, x]

def logsumexp(x, axis=0):
    m = x.max(axis=axis)
    return m + np.log(np.exp(x - np.expand_dims(m, axis)).sum(axis=axis))

def forward(pi, A, B, observations):
    T, K = len(observations), len(pi)

    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(pi) + np.log(emission(B, observations[0]))

    for t in range(1, T):
        e = np.log(emission(B, observations[t]))

        for j in range(K):
            log_alpha[t, j] = logsumexp(log_alpha[t - 1] + np.log(A[:, j])) + e[j]

    return log_alpha

def backward(A, B, observations):
    T, K = len(observations), A.shape[0]

    log_beta = np.zeros((T, K))

    for t in range(T - 2, -1, -1):
        scores = np.log(A) + np.log(emission(B, observations[t + 1])) + log_beta[t + 1]
        log_beta[t] = logsumexp(scores, axis=1)

    return log_beta

def posteriors(pi, A, B, observations):
    T, K = len(observations), A.shape[0]

    log_alpha = forward(pi, A, B, observations)
    log_beta = backward(A, B, observations)
    log_p = logsumexp(log_alpha[-1])

    gamma = np.exp(log_alpha + log_beta - log_p)

    xi_sum = np.zeros_like(A)
    e = [np.log(emission(B, x)) for x in observations]
    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                log_xi = log_alpha[t, i] + np.log(A[i, j]) + e[t + 1][j] + log_beta[t + 1, j] - log_p
                xi_sum[i, j] += np.exp(log_xi)

    return gamma, xi_sum, log_p
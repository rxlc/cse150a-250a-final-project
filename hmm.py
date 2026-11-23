import numpy as np

def emission(B, x):
    """b_i(x) for every state i, returns all ones if x is -1 (missing observation)"""
    if x < 0:
        return np.ones(B.shape[0])
    return B[:, x]

def logsumexp(x, axis=0):
    """log(sum(exp(x))) along an axis"""
    m = x.max(axis=axis)
    return m + np.log(np.exp(x - np.expand_dims(m, axis)).sum(axis=axis))

def forward(pi, A, B, observations):
    """log alpha_t(i) = log P(o_1:t, S_t = i)"""
    T, K = len(observations), len(pi)

    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(pi) + np.log(emission(B, observations[0]))

    for t in range(1, T):
        e = np.log(emission(B, observations[t]))

        for j in range(K):
            log_alpha[t, j] = logsumexp(log_alpha[t - 1] + np.log(A[:, j])) + e[j]

    return log_alpha

def backward(A, B, observations):
    """log beta_t(i) = log P(o_t+1:T | S_t = i)"""
    T, K = len(observations), A.shape[0]

    log_beta = np.zeros((T, K))

    for t in range(T - 2, -1, -1):
        e = np.log(emission(B, observations[t + 1]))
        for i in range(K):
            log_beta[t, i] = logsumexp(np.log(A[i, :]) + e + log_beta[t + 1])
    
    return log_beta

def posteriors(pi, A, B, observations):
    T, K = len(observations), A.shape[0]

    log_alpha = forward(pi, A, B, observations)
    log_beta = backward(A, B, observations)
    log_p = logsumexp(log_alpha[-1])

    gamma = np.exp(log_alpha + log_beta - log_p)

    xi_sum = np.zeros((K, K))
    e = [np.log(emission(B, x)) for x in observations]
    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                log_xi = log_alpha[t, i] + np.log(A[i, j]) + e[t + 1][j] + log_beta[t + 1, j] - log_p
                xi_sum[i, j] += np.exp(log_xi)

    return gamma, xi_sum, log_p

def e_step(pi, A, B, days, pseudo):
    """Expected counts summed over all days. Every count starts at the pseudo-count so no probability becomes exactly 0"""
    K, M = B.shape
    n_pi = np.full(K, pseudo)
    n_A = np.full((K, K), pseudo)
    n_B = np.full((K, M), pseudo)

    total_log_p = 0.0
    for observations in days:
        gamma, xi_sum, log_p = posteriors(pi, A, B, observations)
        n_pi += gamma[0]
        n_A += xi_sum
        
        for t, x in enumerate(observations):
            if x >= 0:
                n_B[:, x] += gamma[t]
        total_log_p += log_p
    
    return n_pi, n_A, n_B, total_log_p

def m_step(n_pi, n_A, n_B):
    pi = n_pi / n_pi.sum()
    A = n_A / n_A.sum(axis = 1, keepdims=True)
    B = n_B / n_B.sum(axis = 1, keepdims=True)

    return pi, A, B

def log_prior(pi, A, B, pseudo):
    """The extra term the pseudo-counts add: EM never decreases log P + log_prior"""
    return pseudo * (np.log(pi).sum() + np.log(A).sum() + np.log(B).sum())

def random_rows(rng, shape):
    """Random probabilities with each row summing to 1"""
    m = rng.random(shape)
    return m / m.sum(axis=-1, keepdims=True)

def fit_once(days, K, M, rng, pseudo, max_iters, tol):
    """Repeat the E-step and M-step until the likelihood barely changes"""
    pi = random_rows(rng, K)
    A = random_rows(rng, (K, K))
    B = random_rows(rng, (K, M))

    objectives = []
    for i in range(max_iters):
        n_pi, n_A, n_B, log_p = e_step(pi, A, B, days, pseudo)
        objectives.append(log_p + log_prior(pi, A, B, pseudo))

        if len(objectives) > 1:
            assert objectives[-1] >= objectives[-2] - 1e-8
            if objectives[-1] - objectives[-2] < tol * abs(objectives[-1]):
                break

        pi, A, B = m_step(n_pi, n_A, n_B)
    return pi, A, B, log_p, objectives

def sort_states(pi, A, B):
    """Sort states by expected level sum_k k * b_ik, so state 0 is calmest and state K-1 the most disrupted"""
    order = np.argsort(B @ np.arange(B.shape[1]))
    return pi[order], A[order][:, order], B[order]

def baum_welch(days, K=3, M=3, restarts=10, pseudo=0.1, max_iters=500, tol=1e-6, seed=11):
    """Best of several EM runs from random starts. Returns pi, A, B (states sorted) and log P of the training days"""
    rng = np.random.default_rng(seed)
    best = None

    for i in range(restarts):
        pi, A, B, log_p, objectives = fit_once(days, K, M, rng, pseudo, max_iters, tol)
        print(f"  restart: log P = {log_p:.2f} after {len(objectives)} iterations")
        if best is None or log_p > best[3]:
            best = (pi, A, B, log_p)

    pi, A, B, log_p = best
    pi, A, B = sort_states(pi, A, B)
    return pi, A, B, log_p

def filter_beliefs(pi, A, B, observations):
    """Detection: P(S_t = i | o_1:t) for every t from the forward pass"""
    log_alpha = forward(pi, A, B, observations)
    beliefs = np.zeros_like(log_alpha)

    for t in range(len(observations)):
        beliefs[t] = np.exp(log_alpha[t] - logsumexp(log_alpha[t]))
    return beliefs

def predict(belief, A, B, k):
    """Prediction k steps ahead: P(S_t+k | o_1:t) and P(O_t+k | o_1:t)"""
    state_dist = belief @ np.linalg.matrix_power(A, k)
    obs_dist = state_dist @ B

    return state_dist, obs_dist

def forecasts(pi, A, B, observations, k):
    """For every t, the predicted distribution of o_t+k made at time t. Row t forecasts observations[t + k]"""
    beliefs = filter_beliefs(pi, A, B, observations)
    Ak_B = np.linalg.matrix_power(A, k) @ B

    return beliefs[: len(observations) - k] @ Ak_B

def viterbi(pi, A, B, observations):
    """Most likely hidden state sequence s_1:T given o_1:T, in log space (described in words in section 3.3)."""
    T, K = len(observations), len(pi)
    log_delta = np.zeros((T, K))

    backpointers_m = np.zeros((T, K), dtype=int)
    log_delta[0] = np.log(pi) + np.log(emission(B, observations[0]))
    for t in range(1, T):
        e = np.log(emission(B, observations[t]))
        
        for j in range(K):
            scores = log_delta[t - 1] + np.log(A[:, j])
            backpointers_m[t, j] = np.argmax(scores)
            log_delta[t, j] = scores[backpointers_m[t, j]] + e[j]

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(log_delta[-1])
    for t in range(T - 1, 0, -1):
        path[t - 1] = backpointers_m[t, path[t]]

    return path, log_delta[-1].max()
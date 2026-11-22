"""Tests for hmm.py. Tiny 2-state mock, expected values from enumerating all paths.

Assumes:
  forward(pi, A, B, obs) -> alpha (T, K) unscaled
  backward(pi, A, B, obs) -> beta (T, K) unscaled
  viterbi(pi, A, B, obs) -> path length T
  baum_welch(pi, A, B, obs, n_iter=...) -> (pi, A, B, loglik_history)

If signatures differ, change the wrappers below.
"""

import itertools

import numpy as np
import pytest

try:
    import hmm as _hmm
except ImportError:
    _hmm = None


def forward(pi, A, B, obs):
    return _hmm.forward(pi, A, B, obs)


def backward(pi, A, B, obs):
    return _hmm.backward(pi, A, B, obs)


def viterbi(pi, A, B, obs):
    return _hmm.viterbi(pi, A, B, obs)


def baum_welch(pi, A, B, obs, n_iter=20):
    return _hmm.baum_welch(pi, A, B, obs, n_iter=n_iter)


# 2 states, 3 symbols, short obs
PI = np.array([0.6, 0.4], dtype=float)
A = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=float)
B = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]], dtype=float)
OBS = np.array([0, 1, 2], dtype=int)
T = len(OBS)


def _path_prob(path, pi, A, B, obs):
    p = pi[path[0]] * B[path[0], obs[0]]
    for t in range(1, len(obs)):
        p *= A[path[t - 1], path[t]] * B[path[t], obs[t]]
    return p


def brute_alpha(pi, A, B, obs):
    t_len = len(obs)
    k = len(pi)
    alpha = np.zeros((t_len, k), dtype=float)
    for t in range(t_len):
        for path in itertools.product(range(k), repeat=t + 1):
            alpha[t, path[t]] += _path_prob(path, pi, A, B, obs[: t + 1])
    return alpha


def brute_beta(pi, A, B, obs):
    t_len = len(obs)
    k = len(pi)
    beta = np.zeros((t_len, k), dtype=float)
    for t in range(t_len):
        remaining = t_len - t - 1
        if remaining == 0:
            beta[t, :] = 1.0
            continue
        for i in range(k):
            total = 0.0
            for future in itertools.product(range(k), repeat=remaining):
                p = 1.0
                prev = i
                for step, state in enumerate(future):
                    p *= A[prev, state] * B[state, obs[t + 1 + step]]
                    prev = state
                total += p
            beta[t, i] = total
    return beta


def brute_total(pi, A, B, obs):
    k = len(pi)
    return sum(
        _path_prob(path, pi, A, B, obs)
        for path in itertools.product(range(k), repeat=len(obs))
    )


def brute_viterbi(pi, A, B, obs):
    k = len(pi)
    best_path, best_p = None, -1.0
    for path in itertools.product(range(k), repeat=len(obs)):
        p = _path_prob(path, pi, A, B, obs)
        if p > best_p:
            best_p = p
            best_path = path
    return np.array(best_path, dtype=int)


def require_hmm():
    if _hmm is None:
        pytest.skip("hmm.py not found")


def test_forward_matches_bruteforce():
    require_hmm()
    alpha = np.asarray(forward(PI, A, B, OBS), dtype=float)
    expected = brute_alpha(PI, A, B, OBS)
    assert alpha.shape == expected.shape
    np.testing.assert_allclose(alpha, expected, rtol=1e-10, atol=1e-12)


def test_backward_matches_bruteforce():
    require_hmm()
    beta = np.asarray(backward(PI, A, B, OBS), dtype=float)
    expected = brute_beta(PI, A, B, OBS)
    assert beta.shape == expected.shape
    np.testing.assert_allclose(beta, expected, rtol=1e-10, atol=1e-12)


def test_forward_backward_same_total_probability():
    require_hmm()
    alpha = np.asarray(forward(PI, A, B, OBS), dtype=float)
    beta = np.asarray(backward(PI, A, B, OBS), dtype=float)
    total_from_alpha = alpha[-1].sum()
    totals = (alpha * beta).sum(axis=1)
    expected = brute_total(PI, A, B, OBS)
    assert np.isclose(total_from_alpha, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(totals, expected, rtol=1e-10, atol=1e-12)


def test_filtered_posterior_from_alpha():
    # P(S_t | o_1..o_t) = alpha[t] / sum(alpha[t])
    require_hmm()
    alpha = np.asarray(forward(PI, A, B, OBS), dtype=float)
    expected_alpha = brute_alpha(PI, A, B, OBS)
    for t in range(T):
        post = alpha[t] / alpha[t].sum()
        expected = expected_alpha[t] / expected_alpha[t].sum()
        np.testing.assert_allclose(post, expected, rtol=1e-10, atol=1e-12)
        assert np.isclose(post.sum(), 1.0)


def test_viterbi_returns_best_path():
    require_hmm()
    path = np.asarray(viterbi(PI, A, B, OBS), dtype=int).ravel()
    expected = brute_viterbi(PI, A, B, OBS)
    assert path.shape == expected.shape
    np.testing.assert_array_equal(path, expected)


def test_baum_welch_loglik_nondecreasing_and_rows_normalize():
    require_hmm()
    obs = np.array([0, 1, 2, 1, 0, 2, 1], dtype=int)
    result = baum_welch(PI.copy(), A.copy(), B.copy(), obs, n_iter=15)
    assert len(result) == 4
    pi_hat, A_hat, B_hat, history = result
    pi_hat = np.asarray(pi_hat, dtype=float)
    A_hat = np.asarray(A_hat, dtype=float)
    B_hat = np.asarray(B_hat, dtype=float)
    history = np.asarray(history, dtype=float).ravel()

    assert history.size >= 1
    assert np.all(np.diff(history) >= -1e-8)

    np.testing.assert_allclose(pi_hat.sum(), 1.0, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(A_hat.sum(axis=1), 1.0, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(B_hat.sum(axis=1), 1.0, rtol=1e-8, atol=1e-10)
    assert np.all(pi_hat >= -1e-12)
    assert np.all(A_hat >= -1e-12)
    assert np.all(B_hat >= -1e-12)


def test_works_with_two_and_three_symbols():
    require_hmm()
    B2 = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    obs2 = np.array([0, 1, 0], dtype=int)
    alpha2 = np.asarray(forward(PI, A, B2, obs2), dtype=float)
    np.testing.assert_allclose(alpha2, brute_alpha(PI, A, B2, obs2), rtol=1e-10, atol=1e-12)

    alpha3 = np.asarray(forward(PI, A, B, OBS), dtype=float)
    np.testing.assert_allclose(alpha3, brute_alpha(PI, A, B, OBS), rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

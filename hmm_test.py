"""Tests for hmm.py. Tiny K = 3 state, M = 3 level mock, expected values from enumerating all paths.

hmm.py's signatures (the wrappers below adapt them to what the tests compare):
  forward(pi, A, B, obs)          -> log alpha (T, K)
  backward(A, B, obs)             -> log beta (T, K)
  filter_beliefs(pi, A, B, obs)   -> P(S_t | o_1:t), (T, K)
  forecasts(pi, A, B, obs, k)     -> row t = P(o_t+k | o_1:t), (T - k, M)
  viterbi(pi, A, B, obs)          -> (path, log P of the path)
  baum_welch(days, K, M, ..., seed) -> (pi, A, B, log P), states sorted calmest first
-1 in obs means missing: its emission probability is 1 for every state.
"""

import itertools

import numpy as np
import pytest

try:
    import hmm as _hmm
except ImportError:
    _hmm = None

K, M, SEED = 3, 3, 0


def forward(pi, A, B, obs):
    return np.exp(_hmm.forward(pi, A, B, obs))


def backward(pi, A, B, obs):
    return np.exp(_hmm.backward(A, B, obs))


def viterbi(pi, A, B, obs):
    path, _ = _hmm.viterbi(pi, A, B, obs)
    return path


# 3 states, 3 levels, short obs
PI = np.array([0.5, 0.3, 0.2], dtype=float)
A = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]], dtype=float)
B = np.array([[0.8, 0.15, 0.05], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]], dtype=float)
OBS = np.array([0, 1, 2, 2], dtype=int)
OBS_MISSING = np.array([0, -1, 2, -1, 1], dtype=int)
T = len(OBS)


def _b(B, state, o):
    """Emission probability, 1 for a missing observation."""
    return 1.0 if o < 0 else B[state, o]


def _path_prob(path, pi, A, B, obs):
    p = pi[path[0]] * _b(B, path[0], obs[0])
    for t in range(1, len(obs)):
        p *= A[path[t - 1], path[t]] * _b(B, path[t], obs[t])
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
                    p *= A[prev, state] * _b(B, state, obs[t + 1 + step])
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
    return np.array(best_path, dtype=int), best_p


def brute_forecast(pi, A, B, obs, t, k):
    """P(o_t+k = m | o_1:t) for every level m: P(o_1:t, o_t+k = m) / P(o_1:t), with the
    steps in between left missing."""
    seen = list(obs[: t + 1])
    joint = np.array([brute_total(pi, A, B, seen + [-1] * (k - 1) + [m]) for m in range(B.shape[1])])
    return joint / brute_total(pi, A, B, seen)


def require_hmm():
    if _hmm is None:
        pytest.skip("hmm.py not found")


@pytest.mark.parametrize("obs", [OBS, OBS_MISSING])
def test_forward_matches_bruteforce(obs):
    require_hmm()
    alpha = forward(PI, A, B, obs)
    expected = brute_alpha(PI, A, B, obs)
    assert alpha.shape == expected.shape
    np.testing.assert_allclose(alpha, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("obs", [OBS, OBS_MISSING])
def test_backward_matches_bruteforce(obs):
    require_hmm()
    beta = backward(PI, A, B, obs)
    expected = brute_beta(PI, A, B, obs)
    assert beta.shape == expected.shape
    np.testing.assert_allclose(beta, expected, rtol=1e-10, atol=1e-12)


def test_forward_backward_same_total_probability():
    require_hmm()
    alpha = forward(PI, A, B, OBS)
    beta = backward(PI, A, B, OBS)
    total_from_alpha = alpha[-1].sum()
    totals = (alpha * beta).sum(axis=1)
    expected = brute_total(PI, A, B, OBS)
    assert np.isclose(total_from_alpha, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(totals, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("obs", [OBS, OBS_MISSING])
def test_filter_beliefs_match_bruteforce(obs):
    # P(S_t | o_1..o_t) = alpha[t] / sum(alpha[t])
    require_hmm()
    beliefs = _hmm.filter_beliefs(PI, A, B, obs)
    expected_alpha = brute_alpha(PI, A, B, obs)
    for t in range(len(obs)):
        expected = expected_alpha[t] / expected_alpha[t].sum()
        np.testing.assert_allclose(beliefs[t], expected, rtol=1e-10, atol=1e-12)
        assert np.isclose(beliefs[t].sum(), 1.0)


@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("obs", [OBS, OBS_MISSING])
def test_forecasts_match_bruteforce(obs, k):
    require_hmm()
    pred = _hmm.forecasts(PI, A, B, obs, k)
    assert pred.shape == (len(obs) - k, M)
    for t in range(len(obs) - k):
        np.testing.assert_allclose(pred[t], brute_forecast(PI, A, B, obs, t, k), rtol=1e-10, atol=1e-12)
        assert np.isclose(pred[t].sum(), 1.0)


@pytest.mark.parametrize("obs", [OBS, OBS_MISSING])
def test_viterbi_returns_best_path(obs):
    require_hmm()
    path, log_p = _hmm.viterbi(PI, A, B, obs)
    _, expected_p = brute_viterbi(PI, A, B, obs)
    # compare probabilities, not paths: with missing steps two paths can tie
    assert len(path) == len(obs)
    assert np.isclose(_path_prob(path, PI, A, B, obs), expected_p, rtol=1e-10)
    assert np.isclose(log_p, np.log(expected_p), rtol=1e-10)


def test_baum_welch_loglik_nondecreasing_and_rows_normalize():
    require_hmm()
    days = [np.array([0, 0, 1, 2, 2, 1, 0, 0], dtype=int),
            np.array([0, 1, -1, 2, 2, 0, 0], dtype=int)]

    # one EM run: the objective (log P + pseudo-count term) never goes down
    _, _, _, _, history = _hmm.fit_once(days, K, M, np.random.default_rng(SEED),
                                        pseudo=0.1, max_iters=50, tol=1e-6)
    assert len(history) >= 1
    assert np.all(np.diff(history) >= -1e-8)

    pi_hat, A_hat, B_hat, log_p = _hmm.baum_welch(days, K=K, M=M, restarts=3, seed=SEED)
    assert pi_hat.shape == (K,) and A_hat.shape == (K, K) and B_hat.shape == (K, M)
    assert np.isfinite(log_p)

    np.testing.assert_allclose(pi_hat.sum(), 1.0, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(A_hat.sum(axis=1), 1.0, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(B_hat.sum(axis=1), 1.0, rtol=1e-8, atol=1e-10)
    # pseudo-counts keep every probability above 0
    assert np.all(pi_hat > 0) and np.all(A_hat > 0) and np.all(B_hat > 0)
    # states sorted calmest first (expected level increasing)
    assert np.all(np.diff(B_hat @ np.arange(M)) >= 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

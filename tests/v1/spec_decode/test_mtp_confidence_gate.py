# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the MTP confidence gate (VLLM_MTP_CONFIDENCE_THRESHOLD)."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.spec_decode.dynamic.utils import (
    batch_size_schedule_depths,
    build_dynamic_sd_schedule_lookup,
    gated_num_spec_tokens,
    mtp_confidence_gate,
    mtp_confidence_gate_max_batch_size,
)
from vllm.v1.worker.gpu.spec_decode.draft_confidence import confident_run_length


def _spec(**overrides):
    fields = dict(
        method="mtp",
        num_speculative_tokens=6,
        num_speculative_tokens_per_batch_size=[(1, 2, 6), (3, 8, 4)],
        draft_sample_method="probabilistic",
        adaptive=False,
    )
    fields.update(overrides)
    adaptive = fields.pop("adaptive")
    return SimpleNamespace(**fields, uses_acceptance_length_adaptation=lambda: adaptive)


@pytest.fixture
def gate_env(monkeypatch):
    def set_env(threshold="0.7", base="4"):
        monkeypatch.setenv("VLLM_MTP_CONFIDENCE_THRESHOLD", threshold)
        monkeypatch.setenv("VLLM_MTP_CONFIDENCE_BASE_DEPTH", base)

    return set_env


def test_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_MTP_CONFIDENCE_THRESHOLD", raising=False)
    assert mtp_confidence_gate(_spec()) is None
    # Off even with an unusable config: nothing is validated.
    assert mtp_confidence_gate(None) is None


def test_gate_on(gate_env):
    gate_env("0.7", "4")
    assert mtp_confidence_gate(_spec()) == (0.7, 4)


@pytest.mark.parametrize(
    "threshold,base,spec,match",
    [
        ("1.5", "4", _spec(), "<= 1"),
        ("0.7", "4", None, "speculative decoding"),
        ("0.7", "4", _spec(method="eagle"), "method='mtp'"),
        ("0.7", "4", _spec(num_speculative_tokens_per_batch_size=None), "per_batch"),
        ("0.7", "4", _spec(adaptive=True), "set one"),
        ("0.7", "4", _spec(draft_sample_method="greedy"), "probabilistic"),
        ("0.7", "6", _spec(), "BASE_DEPTH=6"),
        ("0.7", "0", _spec(), "BASE_DEPTH=0"),
    ],
)
def test_gate_rejects_bad_setups(gate_env, threshold, base, spec, match):
    gate_env(threshold, base)
    with pytest.raises(ValueError, match=match):
        mtp_confidence_gate(spec)


def test_gated_depth():
    # Schedule depth at or below base: the gate does nothing.
    assert gated_num_spec_tokens(4, 4, [False]) == 4
    assert gated_num_spec_tokens(3, 4, [False]) == 3
    # Above base: open only when every request's chain held.
    assert gated_num_spec_tokens(6, 4, [True]) == 6
    assert gated_num_spec_tokens(6, 4, [True, True]) == 6
    assert gated_num_spec_tokens(6, 4, [True, False]) == 4
    assert gated_num_spec_tokens(6, 4, iter([False])) == 4


@pytest.mark.parametrize(
    "schedule,expected",
    [
        ([(1, 2, 6), (3, 8, 4)], 2),
        ([(1, 8, 6)], 8),
        ([(1, 8, 4)], 0),
        # Gaps carry the previous depth forward: 3-4 keep 6.
        ([(1, 2, 6), (5, 8, 4)], 4),
        # Larger than max_num_seqs clips.
        ([(1, 64, 6)], 8),
    ],
)
def test_gate_max_batch_size(schedule, expected):
    dense = build_dynamic_sd_schedule_lookup(schedule, 8, 6)
    assert mtp_confidence_gate_max_batch_size(dense, 4) == expected


def test_schedule_depths_include_gate_base(gate_env, monkeypatch):
    spec = _spec()
    monkeypatch.delenv("VLLM_MTP_CONFIDENCE_THRESHOLD", raising=False)
    assert batch_size_schedule_depths(spec) == [4, 6]
    assert batch_size_schedule_depths(
        _spec(num_speculative_tokens_per_batch_size=[(1, 8, 6)])
    ) == [6]
    gate_env("0.9", "4")
    assert batch_size_schedule_depths(
        _spec(num_speculative_tokens_per_batch_size=[(1, 8, 6)])
    ) == [4, 6]
    # Depths above num_speculative_tokens clamp.
    assert batch_size_schedule_depths(
        _spec(num_speculative_tokens_per_batch_size=[(1, 2, 9), (3, 8, 4)])
    ) == [4, 6]


def test_confident_run_length():
    q = torch.tensor(
        [
            [0.9, 0.8, 0.75, 0.71],  # all confident
            [0.9, 0.5, 0.95, 0.99],  # stops at the second draft
            [0.1, 0.9, 0.9, 0.9],  # stops at the first
            [0.7, 0.7, 0.7, 0.7],  # threshold is inclusive
        ]
    )
    assert confident_run_length(q, 0.7).tolist() == [4, 1, 0, 4]
    assert confident_run_length(q[:, :0], 0.7).tolist() == [0, 0, 0, 0]


def test_threshold_hashed_only_when_on(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_MTP_CONFIDENCE_THRESHOLD", raising=False)
    monkeypatch.setenv("VLLM_MTP_CONFIDENCE_BASE_DEPTH", "3")
    off = envs.compile_factors()
    assert "VLLM_MTP_CONFIDENCE_THRESHOLD" not in off
    # The base depth only matters while the gate is on.
    assert "VLLM_MTP_CONFIDENCE_BASE_DEPTH" not in off

    monkeypatch.setenv("VLLM_MTP_CONFIDENCE_THRESHOLD", "0.7")
    on = envs.compile_factors()
    assert on["VLLM_MTP_CONFIDENCE_THRESHOLD"] is True
    assert on["VLLM_MTP_CONFIDENCE_BASE_DEPTH"] == 3
    # A threshold sweep shares one compile key; a base-depth change does not.
    monkeypatch.setenv("VLLM_MTP_CONFIDENCE_THRESHOLD", "0.8")
    assert envs.compile_factors() == on
    monkeypatch.setenv("VLLM_MTP_CONFIDENCE_BASE_DEPTH", "2")
    assert envs.compile_factors() != on


# --------------------------------------------------------------------------
# Exactness: speculative sampling with a confidence-chosen depth reproduces the
# target distribution. Toy Markov target p(x | prev) and drafter q(x | prev) over
# V tokens; generate L tokens with draft rounds whose depth comes from the gate
# (lagged: last round's chain) and, as a second policy, with an in-round stop
# (TensorFold style: no further drafts after the first q < threshold). Compare the
# L-token sequence histogram with the exact joint by chi-square, as the published
# engines' rejection-sampler tests do.
# --------------------------------------------------------------------------

V, L, BASE, MAX_DEPTH, TAU = 4, 3, 1, 3, 0.5


def _toy_chains(rng):
    p = rng.dirichlet(np.full(V, 0.6), size=V + 1)  # row V = start state
    q = rng.dirichlet(np.full(V, 0.6), size=V + 1)
    # Make the drafter close to the target on some states, far on others.
    q[: V // 2] = 0.8 * p[: V // 2] + 0.2 * q[: V // 2]
    return p, q


def _generate(rng, p, q, policy):
    out: list[int] = []
    prev = V
    confident = False
    while len(out) < L:
        depth = gated_num_spec_tokens(MAX_DEPTH, BASE, [confident])
        drafts, probs = [], []
        state = prev
        for _ in range(depth):
            x = rng.choice(V, p=q[state])
            drafts.append(x)
            probs.append(q[state][x])
            state = x
            if policy == "in_round_stop" and probs[-1] < TAU:
                break
        # Lagged gate input: did the whole chain stay confident?
        confident = all(pr >= TAU for pr in probs)
        # Verify (Leviathan et al.): accept x with prob min(1, p/q), else
        # resample from norm(max(p - q, 0)); all accepted -> bonus from p.
        state = prev
        for x in drafts:
            if rng.random() < min(1.0, p[state][x] / q[state][x]):
                out.append(x)
                state = x
                continue
            residual = np.maximum(p[state] - q[state], 0.0)
            out.append(rng.choice(V, p=residual / residual.sum()))
            break
        else:
            out.append(rng.choice(V, p=p[state]))
        prev = out[-1]
    return tuple(out[:L])


@pytest.mark.parametrize("policy", ["lagged_gate", "in_round_stop"])
def test_confidence_depth_keeps_target_distribution(policy):
    rng = np.random.default_rng(1234)
    p, q = _toy_chains(rng)
    n = 30000
    counts: dict[tuple[int, ...], int] = {}
    for _ in range(n):
        seq = _generate(rng, p, q, policy)
        counts[seq] = counts.get(seq, 0) + 1

    chi2, df = 0.0, -1
    for seq in itertools.product(range(V), repeat=L):
        prob, state = 1.0, V
        for x in seq:
            prob *= p[state][x]
            state = x
        expected = prob * n
        if expected < 5:
            continue
        chi2 += (counts.get(seq, 0) - expected) ** 2 / expected
        df += 1
    # ~10 sigma under the normal approximation, as in
    # tests/v1/spec_decode/test_rejection_sampler_utils.py.
    assert chi2 < df + 10 * (2 * df) ** 0.5, f"{policy}: chi2={chi2:.1f} df={df}"


def test_biased_verifier_is_caught():
    """The chi-square above has power: accepting every draft fails it."""
    rng = np.random.default_rng(7)
    p, q = _toy_chains(rng)
    n = 30000
    counts: dict[tuple[int, ...], int] = {}
    for _ in range(n):
        seq, state = [], V
        for _ in range(L):
            state = rng.choice(V, p=q[state])
            seq.append(state)
        counts[tuple(seq)] = counts.get(tuple(seq), 0) + 1
    chi2, df = 0.0, -1
    for seq in itertools.product(range(V), repeat=L):
        prob, state = 1.0, V
        for x in seq:
            prob *= p[state][x]
            state = x
        if prob * n < 5:
            continue
        chi2 += (counts.get(seq, 0) - prob * n) ** 2 / (prob * n)
        df += 1
    assert chi2 > df + 10 * (2 * df) ** 0.5


# --------------------------------------------------------------------------
# Scheduler: the depth for the next round follows the reported confident runs.
# --------------------------------------------------------------------------


def _gate_scheduler(monkeypatch):
    from tests.v1.core.utils import create_scheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.structured_output import StructuredOutputManager

    monkeypatch.delenv("VLLM_MTP_CONFIDENCE_THRESHOLD", raising=False)
    try:
        base = create_scheduler(
            max_num_seqs=8, max_num_batched_tokens=8192, num_speculative_tokens=6
        )
    except OSError as e:  # model config needs the HF hub or its cache
        pytest.skip(f"no model config: {e}")
    # The helper builds an ngram config; switch it to a gated MTP one.
    monkeypatch.setenv("VLLM_MTP_CONFIDENCE_THRESHOLD", "0.7")
    monkeypatch.setenv("VLLM_MTP_CONFIDENCE_BASE_DEPTH", "4")
    spec = base.vllm_config.speculative_config
    spec.num_speculative_tokens_per_batch_size = [(1, 2, 6), (3, 8, 4)]
    spec.method = "mtp"
    spec.draft_sample_method = "probabilistic"
    return Scheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        block_size=base.block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(base.vllm_config),
    )


def test_scheduler_follows_confident_runs(monkeypatch):
    from tests.v1.core.utils import create_requests
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

    scheduler = _gate_scheduler(monkeypatch)
    assert scheduler.confidence_gate_base_depth == 4
    [request] = create_requests(num_requests=1, num_tokens=8, max_tokens=64)
    scheduler.add_request(request)
    rid = request.request_id

    def output(sampled, confident=None):
        return ModelRunnerOutput(
            req_ids=[rid],
            req_id_to_index={rid: 0},
            sampled_token_ids=[sampled],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            num_confident_draft_tokens=confident,
        )

    # Prefill: no verified chain yet, so the drafter is told base depth.
    out = scheduler.schedule()
    assert out.num_spec_tokens_to_schedule == 4
    scheduler.update_from_output(out, output([0]))
    scheduler.update_draft_token_ids(DraftTokenIds([rid], [[1, 2, 3, 4]]))

    # Verify those 4; still no verified chain when this step is scheduled.
    out = scheduler.schedule()
    assert len(out.scheduled_spec_decode_tokens[rid]) == 4
    assert out.num_spec_tokens_to_schedule == 4
    # All 4 drafts were confident: the next round opens to the schedule's 6.
    scheduler.update_from_output(out, output([1, 2, 3, 4, 5], confident=[4]))
    scheduler.update_draft_token_ids(DraftTokenIds([rid], [[6, 7, 8, 9]]))
    out = scheduler.schedule()
    assert out.num_spec_tokens_to_schedule == 6

    # This chain stopped at its 4th draft: back to base depth.
    scheduler.update_from_output(out, output([6, 7], confident=[3]))
    scheduler.update_draft_token_ids(DraftTokenIds([rid], [[1, 2, 3, 4, 5, 6]]))
    out = scheduler.schedule()
    assert len(out.scheduled_spec_decode_tokens[rid]) == 6
    assert out.num_spec_tokens_to_schedule == 4

    # A full 6-draft chain keeps it open.
    scheduler.update_from_output(out, output([1, 2, 3], confident=[6]))
    scheduler.update_draft_token_ids(DraftTokenIds([rid], [[1, 2, 3, 4]]))
    assert scheduler.schedule().num_spec_tokens_to_schedule == 6


def test_weights_stage_plans_every_schedule_depth(monkeypatch):
    """Depth-4 verify sizes need exact-M b12x plans next to the depth-6 ones.

    k51: without them 442 dense layers served M = 5/10/20/25 from the
    capacity regime and every depth-4 step was 10-16 ms slower than v3d.
    """
    from vllm.model_executor.warmup.b12x_prepare import _planned_decode_counts

    monkeypatch.delenv("VLLM_MTP_CONFIDENCE_THRESHOLD", raising=False)
    capture = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80)

    def counts(schedule):
        spec = _spec(num_speculative_tokens_per_batch_size=schedule)
        spec.enable_adaptive_verification = False
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(decode_query_len=7),
            scheduler_config=SimpleNamespace(max_num_seqs=8),
            vllm_config=SimpleNamespace(
                compilation_config=SimpleNamespace(max_cudagraph_capture_size=80),
                speculative_config=spec,
            ),
        )
        return set(
            _planned_decode_counts(worker, capture_sizes=capture, speculative_tokens=6)
        )

    with_schedule = counts([(1, 2, 6), (3, 8, 4)])
    assert {5, 10, 20, 25, 40} <= with_schedule  # depth 4 at c1, c2, c4, c5, c8
    assert {7, 14, 28, 35, 56} <= with_schedule  # depth 6
    without = counts(None)
    assert {5, 10, 20, 25} & without == set()

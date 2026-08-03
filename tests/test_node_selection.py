"""Выбор узла: порядок предпочтений при нескольких GPU."""

from __future__ import annotations

import pytest

from images_api.services.node_pool import NodePool, NodeRuntime


def node(name, *, free_mb, resident=None, busy=False, models=("z-image-turbo", "hidream-o1"), weight=100):
    n = NodeRuntime(
        name=name,
        kind="local",
        base_urls={"zimage": "http://x", "hidream": "http://y"},
        weight=weight,
    )
    n.gpu_total_mb = 24576
    n.gpu_free_mb = free_mb
    n.vram_reserve_mb = 1024
    n.models = dict.fromkeys(models, True)
    n.resident_model = resident
    n.healthy = True
    n.busy_task_id = "t" if busy else None
    return n


@pytest.fixture
def pool():
    p = NodePool.__new__(NodePool)
    p.nodes = {}
    return p


def test_prefers_node_that_already_holds_the_model(pool):
    """Резидентная модель экономит минуты загрузки — это важнее свободной памяти."""
    pool.nodes = {
        "br-01": node("br-01", free_mb=23000),
        "nv-02": node("nv-02", free_mb=12000, resident="hidream-o1"),
    }
    assert next(n.name for n in pool.candidates("hidream-o1")) == "nv-02"


def test_prefers_more_free_vram_when_neither_holds_it(pool):
    """br-01 идёт первым по алфавиту, но памяти там меньше — выбирать надо nv-02."""
    pool.nodes = {
        "br-01": node("br-01", free_mb=11000),
        "nv-02": node("nv-02", free_mb=23000),
    }
    assert next(n.name for n in pool.candidates("hidream-o1")) == "nv-02"


def test_free_node_beats_busy_one_with_more_vram(pool):
    pool.nodes = {
        "br-01": node("br-01", free_mb=23000, busy=True),
        "nv-02": node("nv-02", free_mb=15000),
    }
    assert next(n.name for n in pool.candidates("hidream-o1")) == "nv-02"


def test_node_without_weights_is_not_a_candidate(pool):
    """«На nv-02 есть hidream, но нет ernie» получается само из /status воркера."""
    pool.nodes = {"nv-02": node("nv-02", free_mb=23000, models=("hidream-o1",))}
    assert pool.candidates("ernie-image-turbo") == []
    assert [n.name for n in pool.candidates("hidream-o1")] == ["nv-02"]


def test_disabled_and_draining_nodes_are_skipped(pool):
    disabled = node("br-01", free_mb=23000)
    disabled.enabled = False
    draining = node("nv-02", free_mb=23000)
    draining.draining = True
    pool.nodes = {"br-01": disabled, "nv-02": draining}
    assert pool.candidates("hidream-o1") == []


def test_unhealthy_node_is_skipped(pool):
    """Узел без достоверной VRAM планировать нельзя."""
    bad = node("nv-02", free_mb=23000)
    bad.healthy = False
    pool.nodes = {"nv-02": bad}
    assert pool.candidates("hidream-o1") == []


def test_forced_node_limits_the_choice(pool):
    pool.nodes = {
        "br-01": node("br-01", free_mb=23000),
        "nv-02": node("nv-02", free_mb=23000),
    }
    assert [n.name for n in pool.candidates("hidream-o1", forced_node="br-01")] == ["br-01"]


def test_usable_mb_subtracts_the_reserve(pool):
    n = node("br-01", free_mb=12000)
    assert n.usable_mb == 12000 - 1024


def test_usable_mb_never_negative(pool):
    n = node("br-01", free_mb=500)
    assert n.usable_mb == 0

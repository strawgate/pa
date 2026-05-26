from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from pa.pydantic_ai_compat import apply_pydantic_ai_v2_harness_compat


def test_run_sync_loop_compat_uses_thread_local_loops():
    apply_pydantic_ai_v2_harness_compat()

    import pydantic_ai._utils as utils

    barrier = Barrier(2)

    def get_loop():
        barrier.wait(timeout=5)
        return utils.get_event_loop()

    with ThreadPoolExecutor(max_workers=2) as pool:
        loops = list(pool.map(lambda _: get_loop(), range(2)))

    assert loops[0] is not loops[1]


def test_combined_toolset_for_run_warning_compat_is_applied():
    apply_pydantic_ai_v2_harness_compat()

    from pydantic_ai.toolsets.combined import CombinedToolset

    assert getattr(CombinedToolset.for_run, "_pa_avoids_unawaited_for_run_warning", False) is True

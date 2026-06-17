"""Run a GeoTrace Temporal worker.

Hosts GeoTraceWorkflow and the GeoTraceActivities bound to a live, bootstrapped
Orchestrator, polling the `geotrace` task queue. Temporal hosts no user code; this
process does. Start a dev server first: `temporal server start-dev` (Web UI on
:8233), then `python -m app.temporal.worker`.

Every workflow and activity must be registered here by reference or it fails at
runtime as "not registered", so the activity list below must track activities.py.
"""

from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from app.config import get_settings
from app.services.orchestrator import Orchestrator
from app.temporal.activities import GeoTraceActivities
from app.temporal.workflows import GeoTraceWorkflow

TASK_QUEUE = "geotrace"


async def main(address: str = "localhost:7233") -> None:
    settings = get_settings()
    orch = await Orchestrator.bootstrap(settings)
    acts = GeoTraceActivities(orch)
    # The pydantic data converter lets QueryIn / PlanGraph / NodeResult round-trip
    # cleanly through history.
    client = await Client.connect(address, data_converter=pydantic_data_converter)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[GeoTraceWorkflow],
        activities=[
            acts.guard,
            acts.plan,
            acts.execute_node,
            acts.summarize,
            acts.output_filter,
            acts.hitl_enqueue,
        ],
    )
    try:
        await worker.run()
    finally:
        await orch.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

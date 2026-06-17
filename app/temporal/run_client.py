"""Start a GeoTrace workflow and print its result.

Usage: python -m app.temporal.run_client "Where could vessel A and B have met?"
Requires a running worker (app.temporal.worker) and a Temporal server.
"""

from __future__ import annotations

import asyncio
import sys
import uuid

from temporalio.client import Client, WorkflowHandle
from temporalio.contrib.pydantic import pydantic_data_converter

from app.models import QueryIn
from app.temporal.models import ReviewDecision
from app.temporal.worker import TASK_QUEUE
from app.temporal.workflows import GeoTraceWorkflow


async def approve(handle: WorkflowHandle, corrected: str | None = None) -> None:
    """What a reviewer or review UI calls to release a parked, low-confidence run."""
    await handle.signal(
        GeoTraceWorkflow.review,
        ReviewDecision(approved=True, corrected_answer=corrected),
    )


async def main(question: str, address: str = "localhost:7233") -> None:
    client = await Client.connect(address, data_converter=pydantic_data_converter)
    handle = await client.start_workflow(
        GeoTraceWorkflow.run,
        QueryIn(question=question),
        id=f"geotrace-{uuid.uuid4().hex[:12]}",
        task_queue=TASK_QUEUE,
    )
    print("started", handle.id)
    print("progress:", await handle.query(GeoTraceWorkflow.progress))
    result = await handle.result()
    print("answer:", result.answer)
    print("confidence:", result.confidence, "hitl_required:", result.hitl_required)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Where could vessel A and vessel B have met?"
    asyncio.run(main(q))

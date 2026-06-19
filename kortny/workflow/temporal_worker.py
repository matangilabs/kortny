"""CLI entrypoint for Kortny's Temporal worker."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from temporalio.client import Client
from temporalio.worker import Worker

from kortny.config import load_settings
from kortny.logging_config import configure_logging
from kortny.workflow.temporal import (
    KortnyTaskWorkflow,
    record_workflow_started_activity,
)

logger = logging.getLogger(__name__)


async def run_temporal_worker(*, once: bool = False) -> None:
    """Run the Temporal worker until interrupted."""

    settings = load_settings()
    client = await Client.connect(
        settings.temporal_address,
        namespace=settings.temporal_namespace,
        plugins=_temporal_plugins(),
    )
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[KortnyTaskWorkflow],
        activities=[record_workflow_started_activity],
    )
    logger.info(
        "temporal worker starting address=%s namespace=%s task_queue=%s",
        settings.temporal_address,
        settings.temporal_namespace,
        settings.temporal_task_queue,
    )
    if once:
        async with worker:
            logger.info("temporal worker boot probe succeeded")
            return
    await worker.run()


def _temporal_plugins() -> list[Any]:
    """Return optional Temporal plugins. The Google ADK plugin was retired with
    the ADK runtime (HIG-281); the seam stays for future plugins."""

    return []


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Temporal workflow worker."""

    configure_logging()
    parser = argparse.ArgumentParser(description="Run the Kortny Temporal worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Start the worker briefly and exit after a boot probe",
    )
    args = parser.parse_args(argv)
    asyncio.run(run_temporal_worker(once=args.once))


if __name__ == "__main__":
    main()

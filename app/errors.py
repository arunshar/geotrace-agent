"""Stable error codes. Never lose context. Never swallow."""

from __future__ import annotations

from typing import Any


class GeoTraceError(Exception):
    code: str = "geotrace.unknown"
    http_status: int = 500
    message: str = "internal error"

    def __init__(self, message: str | None = None, **context: Any) -> None:
        super().__init__(message or self.message)
        if message:
            self.message = message
        self.context = context


class BudgetExceeded(GeoTraceError):
    code = "geotrace.budget_exceeded"
    http_status = 402


class PlanInfeasible(GeoTraceError):
    code = "geotrace.plan_infeasible"
    http_status = 400


class ToolUnavailable(GeoTraceError):
    code = "geotrace.tool_unavailable"
    http_status = 503


class KinematicViolation(GeoTraceError):
    code = "geotrace.kinematic_violation"
    http_status = 422


class HitlRequired(GeoTraceError):
    code = "geotrace.hitl_required"
    http_status = 202


class GuardrailTripped(GeoTraceError):
    code = "geotrace.guardrail"
    http_status = 400

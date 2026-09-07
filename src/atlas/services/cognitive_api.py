"""The Cognitive API (docs/design/03_system_architecture.md section 3.6:
"Cognitive API (`/playbooks/{id}/approve|reject`, `/incidents/{id}`) -- JWT; the
approval queue backend."). Wraps the real Supervisor LangGraph
(`atlas.cognitive.supervisor.build_supervisor_graph`) behind an HTTP surface a
dashboard (Week 10's React frontend) can actually call -- this is the approval-queue
backend the XAI panel and approval queue UI pieces need to exist before there's
anything to build a frontend against.

Read-only endpoints (list/get an incident) are unauthenticated, matching section 3.6's
"Read-only APIs... are unauthenticated locally, and are the only surface exposed by the
public demo." Mutating endpoints (submit an incident, approve/reject a paused proposal)
require a JWT via `atlas.services.auth.require_operator` -- submission is included
alongside approve/reject even though section 3.6 only names approve/reject explicitly,
because submitting an incident can itself trigger a real AUTO_APPLY playbook action
(docs/design/04_agent_architecture.md section 4.4), the same class of consequence
approve/reject has.

Real gap, documented rather than papered over (same honesty as
`corridor_coordinator.py`'s own "no Redis/state-writer pipeline yet" note): incidents
normally arrive via the `incident.events.v1` bus topic, published by the anomaly
detection service or scenario engine, and a bus-consuming wrapper around this service
would invoke `submit_incident` automatically. That consumer isn't wired up here (no
live Kafka/Redpanda locally, same gap as every other bus-based service in this
project) -- `POST /incidents` is this service's own documented substitute so the API
and a frontend can be built and tested against real Supervisor graph execution today.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from atlas.cognitive.supervisor import SupervisorState, initial_state
from atlas.common.logging import get_logger
from atlas.services.auth import require_operator

logger = get_logger(__name__)


class IncidentNotFoundError(KeyError):
    """No incident with this id has been submitted to this service instance."""


class ApprovalNotPendingError(ValueError):
    """The incident exists but isn't currently paused awaiting operator approval --
    either it already resolved (auto-applied or previously approved/rejected) or
    hasn't reached the decide node yet. Resolving twice, or resolving something that
    auto-applied, is a caller bug this makes impossible to do silently."""


class CognitiveApiService:
    """The service-layer logic behind the Cognitive API, kept separate from the
    FastAPI wiring so it's directly unit-testable against a real compiled Supervisor
    graph with no HTTP layer involved -- the same separation `IntersectionAgentService`/
    `CorridorCoordinatorService` use elsewhere in this project.
    """

    def __init__(
        self, graph: CompiledStateGraph[SupervisorState, None, SupervisorState, SupervisorState]
    ) -> None:
        self._graph = graph
        self._known_incident_ids: set[str] = set()

    @staticmethod
    def _config(incident_id: str) -> RunnableConfig:
        return RunnableConfig(configurable={"thread_id": incident_id})

    def submit_incident(self, incident: Mapping[str, object]) -> dict[str, Any]:
        incident_id = str(incident["incident_id"])
        config = self._config(incident_id)
        self._graph.invoke(initial_state(dict(incident)), config=config)
        self._known_incident_ids.add(incident_id)
        logger.info("cognitive_api.incident_submitted", incident_id=incident_id)
        return self.get_incident(incident_id)

    def get_incident(self, incident_id: str) -> dict[str, Any]:
        if incident_id not in self._known_incident_ids:
            raise IncidentNotFoundError(f"No incident with id '{incident_id}' has been submitted")

        snapshot = self._graph.get_state(self._config(incident_id))
        pending_approval = snapshot.next != ()
        interrupt_payload: dict[str, object] | None = None
        if pending_approval and snapshot.tasks and snapshot.tasks[0].interrupts:
            interrupt_payload = snapshot.tasks[0].interrupts[0].value

        return {
            "incident_id": incident_id,
            "state": dict(snapshot.values),
            "pending_approval": pending_approval,
            "interrupt": interrupt_payload,
        }

    def list_incidents(self) -> list[dict[str, Any]]:
        return [self.get_incident(incident_id) for incident_id in sorted(self._known_incident_ids)]

    def list_pending_approvals(self) -> list[dict[str, Any]]:
        return [incident for incident in self.list_incidents() if incident["pending_approval"]]

    def resolve_approval(self, incident_id: str, approved: bool) -> dict[str, Any]:
        if incident_id not in self._known_incident_ids:
            raise IncidentNotFoundError(f"No incident with id '{incident_id}' has been submitted")

        config = self._config(incident_id)
        snapshot = self._graph.get_state(config)
        if snapshot.next == ():
            raise ApprovalNotPendingError(
                f"Incident '{incident_id}' is not currently awaiting approval "
                f"(status='{snapshot.values.get('status')}')"
            )

        self._graph.invoke(Command(resume=approved), config=config)
        logger.info("cognitive_api.approval_resolved", incident_id=incident_id, approved=approved)
        return self.get_incident(incident_id)


def create_app(service: CognitiveApiService, jwt_secret: str) -> FastAPI:
    app = FastAPI(title="Cognitive API")
    require_jwt = require_operator(jwt_secret)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/incidents")
    def list_incidents() -> list[dict[str, Any]]:
        return service.list_incidents()

    @app.get("/incidents/{incident_id}")
    def get_incident(incident_id: str) -> dict[str, Any]:
        try:
            return service.get_incident(incident_id)
        except IncidentNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    @app.post("/incidents", status_code=status.HTTP_201_CREATED)
    def submit_incident(
        incident: dict[str, Any],
        _claims: dict[str, object] = Depends(require_jwt),  # noqa: B008
    ) -> dict[str, Any]:
        return service.submit_incident(incident)

    @app.post("/incidents/{incident_id}/approve")
    def approve(
        incident_id: str,
        _claims: dict[str, object] = Depends(require_jwt),  # noqa: B008
    ) -> dict[str, Any]:
        try:
            return service.resolve_approval(incident_id, approved=True)
        except IncidentNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ApprovalNotPendingError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.post("/incidents/{incident_id}/reject")
    def reject(
        incident_id: str,
        _claims: dict[str, object] = Depends(require_jwt),  # noqa: B008
    ) -> dict[str, Any]:
        try:
            return service.resolve_approval(incident_id, approved=False)
        except IncidentNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ApprovalNotPendingError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return app

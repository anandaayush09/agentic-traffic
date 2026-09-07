import type { Incident } from "./api";

interface ApprovalQueueProps {
  incidents: Incident[];
  selectedId: string | null;
  onSelectIncident: (incidentId: string) => void;
  onApprove: (incidentId: string) => void;
  onReject: (incidentId: string) => void;
  busyId: string | null;
}

export function ApprovalQueue({
  incidents,
  selectedId,
  onSelectIncident,
  onApprove,
  onReject,
  busyId,
}: ApprovalQueueProps) {
  if (incidents.length === 0) {
    return <p className="empty">No incidents submitted yet.</p>;
  }

  return (
    <ul className="approval-queue">
      {incidents.map((incident) => (
        <li
          key={incident.incident_id}
          className={incident.incident_id === selectedId ? "selected" : ""}
          onClick={() => onSelectIncident(incident.incident_id)}
        >
          <div className="row">
            <span className="incident-id">{incident.incident_id}</span>
            <span className={`status status-${incident.state.status}`}>
              {incident.state.status}
            </span>
          </div>
          <div className="row playbook">
            {incident.state.proposal?.playbook_name ?? "(no proposal yet)"}
          </div>
          {incident.pending_approval && (
            <div className="row actions">
              <button
                type="button"
                disabled={busyId === incident.incident_id}
                onClick={(e) => {
                  e.stopPropagation();
                  onApprove(incident.incident_id);
                }}
              >
                Approve
              </button>
              <button
                type="button"
                disabled={busyId === incident.incident_id}
                onClick={(e) => {
                  e.stopPropagation();
                  onReject(incident.incident_id);
                }}
              >
                Reject
              </button>
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

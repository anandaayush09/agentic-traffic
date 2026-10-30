import { CONTROLLER_MODES, type ControllerMode, type TrafficJunction } from "./api";

interface SignalControlProps {
  junctions: Record<string, TrafficJunction> | undefined;
  /** Without an operator token the panel is look-only. */
  canControl: boolean;
  busyId: string | null;
  onSetMode: (tlsId: string, mode: ControllerMode) => void;
}

const MODE_LABEL: Record<string, string> = {
  fixed_time: "Fixed-time",
  actuated: "Actuated",
  max_pressure: "Max-pressure",
};

export function SignalControl({ junctions, canControl, busyId, onSetMode }: SignalControlProps) {
  const entries = Object.entries(junctions ?? {});
  if (entries.length === 0) {
    return <p className="empty">Waiting for the live simulation…</p>;
  }
  return (
    <>
      <ul className="signal-control">
        {entries.map(([tlsId, junction]) => (
          <li key={tlsId}>
            <span className="junction-id" title={tlsId}>
              …{tlsId.slice(-4)}
            </span>
            <select
              value={junction.mode}
              disabled={!canControl || busyId === tlsId || junction.playbook_override}
              onChange={(e) => onSetMode(tlsId, e.target.value as ControllerMode)}
            >
              {CONTROLLER_MODES.map((mode) => (
                <option key={mode} value={mode}>
                  {MODE_LABEL[mode]}
                </option>
              ))}
            </select>
            <span className="junction-queue">{junction.queue} queued</span>
            {junction.playbook_override && <span className="override-badge">playbook</span>}
          </li>
        ))}
      </ul>
      <p className="empty">
        ATLAS controllers drive these 5 junctions through the safety shield; the other signals
        keep their native plans.
        {canControl ? "" : " Paste an operator token to change a mode."}
      </p>
    </>
  );
}

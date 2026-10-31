import { useState } from "react";
import { DEFAULT_CONTROLLER_MODES, type ControllerMode, type TrafficJunction } from "./api";

interface SignalControlProps {
  junctions: Record<string, TrafficJunction> | undefined;
  /** Modes the API accepts now; `rl` is present only when a trained policy is loaded. */
  availableModes: string[] | undefined;
  /** Without an operator token the panel is look-only. */
  canControl: boolean;
  busyId: string | null;
  onSetMode: (tlsId: string, mode: ControllerMode) => void;
}

const MODE_LABEL: Record<string, string> = {
  fixed_time: "Fixed-time",
  actuated: "Actuated",
  max_pressure: "Max-pressure",
  rl: "RL policy",
};

// A whole city has thousands of signals; a row each would freeze the page. Past this many,
// the panel shows the busiest few and a search box for the rest.
const LIST_LIMIT = 60;
const SHOWN_WHEN_LARGE = 40;

export function SignalControl({
  junctions,
  availableModes,
  canControl,
  busyId,
  onSetMode,
}: SignalControlProps) {
  const [search, setSearch] = useState("");
  const all = Object.entries(junctions ?? {});
  const large = all.length > LIST_LIMIT;
  const needle = search.trim();
  const entries = large
    ? all
        .filter(([id]) => (needle ? id.includes(needle) : true))
        .sort(([, a], [, b]) => Number(b.playbook_override) - Number(a.playbook_override) || b.queue - a.queue)
        .slice(0, SHOWN_WHEN_LARGE)
    : all;
  if (all.length === 0) {
    return <p className="empty">Waiting for the live simulation…</p>;
  }
  const modes = availableModes && availableModes.length > 0 ? availableModes : DEFAULT_CONTROLLER_MODES;
  return (
    <>
      {large && (
        <input
          type="search"
          className="signal-search"
          placeholder={`Search ${all.length} signals by id`}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      )}
      <ul className="signal-control">
        {entries.map(([tlsId, junction]) => (
          <li key={tlsId}>
            <span className="junction-id" title={tlsId}>
              …{tlsId.slice(-4)}
            </span>
            <select
              value={junction.mode}
              disabled={!canControl || busyId === tlsId || junction.playbook_override}
              onChange={(e) => onSetMode(tlsId, e.target.value)}
            >
              {modes.map((mode) => (
                <option key={mode} value={mode}>
                  {MODE_LABEL[mode] ?? mode}
                </option>
              ))}
            </select>
            <span className="junction-queue">{junction.queue} queued</span>
            {junction.playbook_override && <span className="override-badge">playbook</span>}
          </li>
        ))}
      </ul>
      <p className="empty">
        {large ? `Showing ${entries.length} of ${all.length} signals (busiest first). ` : ""}
        ATLAS controllers drive all {all.length} signalized junctions through the safety
        shield.
        {canControl ? "" : " Paste an operator token to change a mode."}
      </p>
    </>
  );
}

import type { TrafficSnapshot } from "./api";

interface ReplayScrubberProps {
  /** False in the static public demo (no live API) or before the simulation is ready. */
  available: boolean;
  frames: TrafficSnapshot[] | null;
  index: number;
  playing: boolean;
  loading: boolean;
  error: string | null;
  onStart: () => void;
  onSeek: (index: number) => void;
  onTogglePlay: () => void;
  onExit: () => void;
}

function formatClock(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function ReplayScrubber({
  available,
  frames,
  index,
  playing,
  loading,
  error,
  onStart,
  onSeek,
  onTogglePlay,
  onExit,
}: ReplayScrubberProps) {
  if (!available) {
    return <p className="empty">Replay needs the live API (it is off in the static demo).</p>;
  }
  if (!frames) {
    return (
      <>
        <button type="button" onClick={onStart} disabled={loading}>
          {loading ? "Loading…" : "Replay the last ~10 minutes"}
        </button>
        {error && <p className="empty">{error}</p>}
        <p className="empty">
          Rewinds the map to recent live frames (kept in memory, about 15 minutes; a restart
          clears them).
        </p>
      </>
    );
  }
  if (frames.length === 0) {
    return (
      <>
        <p className="empty">No frames recorded yet.</p>
        <button type="button" onClick={onExit}>
          Back to live
        </button>
      </>
    );
  }
  const frame = frames[Math.min(index, frames.length - 1)];
  return (
    <div className="replay">
      <input
        type="range"
        min={0}
        max={frames.length - 1}
        value={index}
        onChange={(e) => onSeek(Number(e.target.value))}
        aria-label="Replay position"
      />
      <div className="replay-row">
        <span>
          Sim clock {formatClock(frame.sim_time_s)} · frame {index + 1}/{frames.length}
        </span>
      </div>
      <div className="replay-row">
        <button type="button" onClick={onTogglePlay}>
          {playing ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={onExit}>
          Back to live
        </button>
      </div>
    </div>
  );
}

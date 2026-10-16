export function computePauseTransition({
    previousTimeMs,
    currentTimeMs,
    isPaused,
    jumpThresholdMs,
    noteEndMarginMs,
    timelineStartMs,
    timelineEndMs,
    epsilonMs = 0,
    freezeStartRealMs = 0,
    freezeSongTimeMs = 0,
    pauseThresholdMs = 500,
    nowRealMs = 0,
}) {
    const prev = Number(previousTimeMs);
    const now = Number(currentTimeMs);
    const threshold = Number(jumpThresholdMs);
    const margin = Number(noteEndMarginMs);
    const epsilon = Math.max(0, Number(epsilonMs) || 0);
    const freezeStart = Number.isFinite(freezeStartRealMs) && freezeStartRealMs > 0 ? freezeStartRealMs : 0;
    const freezeSong = Number.isFinite(freezeSongTimeMs) ? freezeSongTimeMs : 0;
    const pauseThreshold = Math.max(0, Number.isFinite(pauseThresholdMs) ? pauseThresholdMs : 500);
    const nowReal = Number.isFinite(nowRealMs) && nowRealMs > 0 ? nowRealMs : 0;

    if (!Number.isFinite(prev) || !Number.isFinite(now) || !Number.isFinite(threshold) || threshold <= 0) {
        return {
            jumped: false,
            atEnd: false,
            sameTime: false,
            nextPaused: Boolean(isPaused),
            shouldAddMarker: false,
            shouldClearMarkers: false,
            frozenInterpMs: null,
            pauseTimeMs: 0,
            freezeStartRealMs: 0,
            freezeSongTimeMs: 0,
        };
    }

    const hasEnd = Number.isFinite(timelineEndMs);
    const atEnd = hasEnd && now >= (Number(timelineEndMs) - (Number.isFinite(margin) ? margin : 0));
    const hasStart = Number.isFinite(timelineStartMs);
    const beforeStart = hasStart && now < Number(timelineStartMs);

    const timeDelta = now - prev;
    const jumped = Math.abs(timeDelta) > threshold && !(timeDelta > 0 && timeDelta < threshold);
    const sameTime = Math.abs(timeDelta) <= epsilon;

    let nextPaused = Boolean(isPaused);
    let shouldAddMarker = false;
    let shouldClearMarkers = false;
    let frozenInterpMs = null;
    let pauseTimeMs = 0;
    let nextFreezeStartRealMs = freezeStart;
    let nextFreezeSongTimeMs = freezeSong;

    if (jumped && !atEnd && !beforeStart) {
        nextPaused = false;
        shouldClearMarkers = true;
        nextFreezeStartRealMs = 0;
        nextFreezeSongTimeMs = 0;
    } else if (sameTime && !atEnd && !beforeStart) {
        if (!nextPaused) {
            if (!freezeStart) {
                nextFreezeStartRealMs = nowReal || freezeStart;
                nextFreezeSongTimeMs = now;
            } else if (nowReal && freezeStart && (nowReal - freezeStart) >= pauseThreshold) {
                nextPaused = true;
                shouldAddMarker = true;
                frozenInterpMs = freezeSong;
                pauseTimeMs = freezeSong;
            }
        }
    } else if (nextPaused) {
        nextPaused = false;
        nextFreezeStartRealMs = 0;
        nextFreezeSongTimeMs = 0;
    } else {
        nextFreezeStartRealMs = 0;
        nextFreezeSongTimeMs = 0;
    }

    return {
        jumped,
        atEnd,
        sameTime,
        nextPaused,
        shouldAddMarker,
        shouldClearMarkers,
        frozenInterpMs,
        pauseTimeMs,
        freezeStartRealMs: nextFreezeStartRealMs,
        freezeSongTimeMs: nextFreezeSongTimeMs,
    };
}

"""Device-selection helpers for the realtime GUI (pure functions, no I/O)."""

# Playback ends of the usual virtual audio cables. Voice chat apps and games
# then use the cable's recording end as their microphone.
VIRTUAL_OUTPUT_HINTS = ("cable input", "voicemeeter input", "voicemeeter aux input")
VIRTUAL_INPUT_HINTS = ("cable output", "voicemeeter out", "voicemeeter aux out")


def find_virtual_cable(output_names):
    """Name of a virtual-cable playback device among ``output_names``, or ''."""
    for hint in VIRTUAL_OUTPUT_HINTS:
        for name in output_names:
            if hint in name.lower():
                return name
    return ""


def is_virtual_input(name):
    """True for the recording end of a virtual cable (not a real microphone)."""
    lowered = name.lower()
    return any(hint in lowered for hint in VIRTUAL_INPUT_HINTS)


def preferred_hostapi(hostapis, windows):
    """WASAPI has far lower latency than Windows' default MME."""
    if windows:
        for name in hostapis:
            if "WASAPI" in name:
                return name
    return hostapis[0] if hostapis else ""


def match_device(name, candidates):
    """The entry of ``candidates`` that is the same physical device as
    ``name`` under another host API. MME truncates names to 31 characters, so
    the comparison is by common prefix."""
    if not name:
        return ""
    if name in candidates:
        return name
    key = name.lower()
    best, best_length = "", 0
    for candidate in candidates:
        other = candidate.lower()
        length = 0
        for a, b in zip(key, other):
            if a != b:
                break
            length += 1
        if length > best_length:
            best, best_length = candidate, length
    # Require most of the shorter name to agree, or it is a different device.
    if best and best_length >= min(len(key), len(best), 20) * 0.8:
        return best
    return ""

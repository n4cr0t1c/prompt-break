"""Obfuscation and normalization utilities (copied from advanced_jailbreak_agent)."""

import base64


LEET_MAP = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
}


def normalize_leetspeak(text):
    return "".join(LEET_MAP.get(ch, ch) for ch in text.lower())


def _safe_ascii(decoded_bytes):
    out = []
    for b in decoded_bytes:
        if 32 <= b <= 126 or b in (9, 10, 13):
            out.append(chr(b))
        else:
            out.append("?")
    return "".join(out)


def detect_base64(text):
    score = 0.0
    previews = []
    reasons = []
    tokens = text.split()
    for token in tokens:
        cleaned = token.strip(".,;:()[]{}<>\"'")
        if len(cleaned) < 16:
            continue
        if len(cleaned) % 4 != 0:
            continue
        allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
        if not all(ch in allowed for ch in cleaned):
            continue
        try:
            raw = base64.b64decode(cleaned, validate=True)
            preview = _safe_ascii(raw[:120]).strip()
            if preview:
                score = max(score, 100.0)
                previews.append(preview)
                reasons.append("Detected likely Base64 payload.")
        except Exception:
            continue
    return score, previews, reasons


def detect_rot13(text):
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    rot = "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"
    trans = str.maketrans(alpha, rot)
    decoded = text.translate(trans)
    markers = ["ignore", "bypass", "system", "instructions", "developer", "safety", "policy"]
    hits = sum(1 for m in markers if m in decoded.lower())
    if hits >= 2 and decoded != text:
        return 70.0, [decoded[:200]], ["Detected suspicious ROT13-like transformed content."]
    return 0.0, [], []


def detect_hex(text):
    score = 0.0
    previews = []
    reasons = []
    compact = "".join(ch for ch in text if ch not in " \n\r\t")
    if len(compact) >= 20 and len(compact) % 2 == 0:
        hex_digits = "0123456789abcdefABCDEF"
        if all(ch in hex_digits for ch in compact):
            try:
                decoded = bytes.fromhex(compact)
                preview = _safe_ascii(decoded[:120]).strip()
                if preview:
                    score = max(score, 85.0)
                    previews.append(preview)
                    reasons.append("Detected likely HEX-encoded payload.")
            except Exception:
                pass
    return score, previews, reasons


def detect_obfuscation(text):
    methods = []
    previews = []
    reasons = []
    best_score = 0.0

    b64_score, b64_previews, b64_reasons = detect_base64(text)
    if b64_score > 0:
        methods.append("base64")
        previews.extend(b64_previews)
        reasons.extend(b64_reasons)
        best_score = max(best_score, b64_score)

    rot_score, rot_previews, rot_reasons = detect_rot13(text)
    if rot_score > 0:
        methods.append("rot13")
        previews.extend(rot_previews)
        reasons.extend(rot_reasons)
        best_score = max(best_score, rot_score)

    hex_score, hex_previews, hex_reasons = detect_hex(text)
    if hex_score > 0:
        methods.append("hex")
        previews.extend(hex_previews)
        reasons.extend(hex_reasons)
        best_score = max(best_score, hex_score)

    decoded_preview = " | ".join(previews[:2]) if previews else ""
    return {
        "score": best_score,
        "methods": methods,
        "decoded_preview": decoded_preview,
        "reasons": reasons,
    }

"""Command-line interface for PROMPT-BREAK.

Provides interactive prompt analysis plus optional ML training/evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any

from .jailbreak_agent import JailbreakAgent  # lightweight, self-contained detector

from .model import MLClassifier, SKLEARN_AVAILABLE


# ── ANSI colours
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    DIM = "\033[2m"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


USE_COLOR = _supports_color()


def c(code: str, text: str) -> str:
    return f"{code}{text}{C.RESET}" if USE_COLOR else text


# ── Banner / UI
_BANNER = f"\n{c(C.CYAN + C.BOLD, '╔' + '═' * 58 + '╗')}\n"
_BANNER += f"{c(C.CYAN + C.BOLD, '║')}  {c(C.WHITE + C.BOLD, 'PROMPT-BREAK')}  {c(C.DIM, '— Interactive Jailbreak Detector')}  {c(C.CYAN + C.BOLD, '║')}\n"
_BANNER += f"{c(C.CYAN + C.BOLD, '╚' + '═' * 58 + '╝')}\n"


def _get_field(result: Any, key: str, default=None):
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _render_result(prompt: str, result: Any) -> str:
    prompt_short = prompt[:80] + ("…" if len(prompt) > 80 else "")
    lines = ["\n" + ("─" * 64)]
    lines.append(c(C.DIM, f"  Prompt : ") + c(C.WHITE, prompt_short))
    lines.append(("─" * 64))

    is_j = _get_field(result, "is_jailbreak_attempt", False)
    if is_j:
        verdict = c(C.RED + C.BOLD, "⚠  JAILBREAK DETECTED")
    else:
        verdict = c(C.GREEN + C.BOLD, "✓  BENIGN")
    lines.append(f"  {verdict}")

    attack_class = _get_field(result, "attack_class", "unknown")
    lines.append(f"  {c(C.DIM, 'Attack Class  :')} {c(C.YELLOW + C.BOLD, str(attack_class))}")

    score = float(_get_field(result, "confidence_score", 0.0) or 0.0)
    bar_len = 30
    filled = int(bar_len * score / 100)
    bar_color = C.RED if score >= 70 else (C.YELLOW if score >= 40 else C.GREEN)
    bar = c(bar_color, "█" * filled) + c(C.DIM, "░" * (bar_len - filled))
    lines.append(f"  {c(C.DIM, 'Confidence    :')} [{bar}] {c(C.BOLD, f'{score:.1f}%')}")

    patterns = _get_field(result, "raw_matched_patterns", []) or _get_field(result, "raw_matched_patterns", [])
    if patterns:
        lines.append(f"  {c(C.DIM, 'Matched Rules :')} {len(patterns)} pattern(s)")
        for pat in patterns[:3]:
            lines.append(f"    {c(C.DIM, '→')} {c(C.DIM, str(pat)[:70])}")

    decoded = _get_field(result, "decoded_preview", "")
    if decoded:
        lines.append(f"  {c(C.DIM, 'Decoded       :')} {c(C.MAGENTA, str(decoded)[:80])}")

    explanation = _get_field(result, "explanation", "No explanation available.")
    lines.append(f"\n  {c(C.DIM, 'Explanation   :')} {explanation}")

    mitigation = _get_field(result, "suggested_mitigation", "")
    if mitigation:
        lines.append(f"\n  {c(C.YELLOW + C.BOLD, '📋 Suggested Mitigation:')}")
        for chunk in [mitigation[i : i + 60] for i in range(0, len(mitigation), 60)]:
            lines.append(f"  {c(C.WHITE, chunk)}")

    lines.append("\n" + ("─" * 64))
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PROMPT-BREAK — CLI jailbreak detector")
    p.add_argument("--once", action="store_true", help="Read one prompt from stdin and exit")
    p.add_argument("--json", action="store_true", dest="json_output", help="Output JSON result")
    p.add_argument(
        "--mode",
        choices=["heuristic", "ml", "hybrid"],
        default="hybrid",
        help="Detection mode: heuristic (rules), ml (trained), hybrid (blend)",
    )
    p.add_argument("--embeddings", action="store_true", help="Use sentence-transformers embeddings for semantic similarity (optional)")
    p.add_argument("--cluster", action="store_true", help="Enable exemplar clustering to suggest attack class (optional)")
    p.add_argument("--gradio", action="store_true", help="Launch Gradio web UI and exit")
    p.add_argument("--train", action="store_true", help="Train an ML classifier on synthetic data")
    p.add_argument("--train-data", type=str, dest="train_data", help="Path to labeled dataset (CSV or JSONL) to use for training")
    p.add_argument("--progress", action="store_true", help="Show progress during dataset processing and training")
    p.add_argument("--absolute", action="store_true", help="Output absolute model predictions for evaluation set when training with dataset")
    p.add_argument("--eval", action="store_true", help="Evaluate or print last-training metrics if available")
    p.add_argument("--model-path", default="prompt_break/models/classifier.joblib", help="Path to persist/load ML model")
    p.add_argument("--threshold", type=float, default=0.5, help="ML probability threshold (0-1)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    agent = JailbreakAgent()

    # Optional semantic engine (sentence-transformers) and clusterer
    semantic_engine = None
    clusterer = None

    ml = MLClassifier(model_path=args.model_path)
    if args.embeddings:
        try:
            from .semantic import SemanticEngine

            semantic_engine = SemanticEngine()
            if not semantic_engine.available:
                print("sentence-transformers not available — will use TF-IDF fallback if present.")
        except Exception:
            semantic_engine = None
    if args.cluster:
        try:
            from .cluster import Clusterer

            clusterer = Clusterer()
            # fit clusterer on available exemplars if semantic engine available
            try:
                from jailbreak_agent import EXEMPLARS

                if semantic_engine:
                    clusterer.fit(EXEMPLARS, semantic_engine)
            except Exception:
                pass
        except Exception:
            clusterer = None
    if args.train or args.train_data:
        if not SKLEARN_AVAILABLE:
            print("ML training requested but scikit-learn is not installed. Install dependencies in requirements.txt to enable ML mode.")
            return
        if args.train_data:
            print(f"Training ML classifier using dataset: {args.train_data}")
            res = ml.train(dataset_path=args.train_data, show_progress=args.progress, output_predictions=args.absolute)
        else:
            print("Training ML classifier on synthetic dataset (this may take a moment)...")
            res = ml.train()
        # `res` is a dict with 'metrics' (and optionally 'predictions')
        metrics = res.get("metrics", res) if isinstance(res, dict) else res
        print("Training complete — evaluation on held-out split:")
        print(json.dumps(metrics, indent=2))
        if args.absolute and isinstance(res, dict) and "predictions" in res:
            print("\nAbsolute predictions (evaluation set):")
            for p in res["predictions"]:
                print(json.dumps(p, ensure_ascii=False))
        return

    if args.eval:
        if not SKLEARN_AVAILABLE:
            print("ML evaluate requested but scikit-learn is not installed.")
            return
        try:
            ml.load()
            print(f"Loaded model from {args.model_path}")
            print("Model ready — use --mode ml or --mode hybrid to apply it.")
        except Exception as e:
            print(f"Model not available: {e}")
        return

    # Launch Gradio UI if requested
    if args.gradio:
        try:
            from .gradio_app import launch_gradio

            launch_gradio(lambda p: analyze_prompt(p))
        except Exception as e:
            print(f"Failed to launch Gradio: {e}")
        return

    if not args.json_output:
        print(_BANNER)

    def analyze_prompt(prompt: str) -> dict:
        heur = agent.analyze(prompt)
        heur_conf = float(heur.get("confidence_score", 0.0))

        if args.mode == "heuristic":
            return heur

        # ML modes
        if not SKLEARN_AVAILABLE:
            # fallback
            return heur

        try:
            ml.load()
            ml_prob = ml.predict_proba([prompt])[0]
        except Exception:
            return heur

        ml_conf_pct = ml_prob * 100.0
        if args.mode == "ml":
            is_jail, _ = ml.predict(prompt, threshold=args.threshold)
            if is_jail:
                # use heuristics to label the attack class / mitigation
                merged = {
                    "is_jailbreak_attempt": True,
                    "attack_class": heur.get("attack_class", "unknown"),
                    "confidence_score": ml_conf_pct,
                    "suggested_mitigation": heur.get("suggested_mitigation", ""),
                    "explanation": f"ML model probability={ml_prob:.3f}",
                    "raw_matched_patterns": heur.get("raw_matched_patterns", []),
                    "decoded_preview": heur.get("decoded_preview", ""),
                }
            else:
                merged = {
                    "is_jailbreak_attempt": False,
                    "attack_class": "benign",
                    "confidence_score": ml_conf_pct,
                    "suggested_mitigation": heur.get("suggested_mitigation", ""),
                    "explanation": f"ML model probability={ml_prob:.3f}",
                    "raw_matched_patterns": [],
                    "decoded_preview": "",
                }
            return merged
        # hybrid: average available signals (heuristic, ml, embeddings)
        sem_pct = 0.0
        sem_label = None
        if semantic_engine:
            try:
                sem_label, sem_score, sem_ex = semantic_engine.most_similar_to_exemplars(prompt)
                sem_pct = float(sem_score * 100.0)
            except Exception:
                sem_pct = 0.0

        heur_w = 0.5
        ml_w = 0.3 if SKLEARN_AVAILABLE else 0.0
        sem_w = 0.2 if semantic_engine else 0.0
        total_w = heur_w + ml_w + sem_w
        if total_w <= 0:
            fused_conf = heur_conf
        else:
            fused_conf = (heur_conf * heur_w + ml_conf_pct * ml_w + sem_pct * sem_w) / total_w
        is_jail = fused_conf >= 50.0
        merged = {
            "is_jailbreak_attempt": bool(is_jail),
            "attack_class": heur.get("attack_class", "unknown") if is_jail else "benign",
            "confidence_score": fused_conf,
            "suggested_mitigation": heur.get("suggested_mitigation", ""),
            "explanation": f"Heuristic={heur_conf:.1f}, ML={ml_conf_pct:.1f}",
            "raw_matched_patterns": heur.get("raw_matched_patterns", []),
            "decoded_preview": heur.get("decoded_preview", ""),
        }
        return merged

    if args.once:
        prompt = sys.stdin.read().strip()
        result = analyze_prompt(prompt)
        if args.json_output:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(_render_result(prompt, result))
        return

    # interactive loop
    print(c(C.DIM, "  Type a prompt and press Enter to analyse. Type 'quit' to exit."))
    while True:
        try:
            prompt = input(c(C.CYAN + C.BOLD, "  ❯ Enter prompt: ")).strip()
        except (KeyboardInterrupt, EOFError):
            print(c(C.DIM, "\n\n  Exiting. Stay safe."))
            break

        if prompt.lower() in {"quit", "exit", "q"}:
            print(c(C.DIM, "\n  Exiting. Stay safe."))
            break

        if not prompt:
            print(c(C.DIM, "  (empty input — please type something)\n"))
            continue

        result = analyze_prompt(prompt)
        if args.json_output:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(_render_result(prompt, result))


if __name__ == "__main__":
    main()

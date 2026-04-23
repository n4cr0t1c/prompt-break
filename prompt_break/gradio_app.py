"""Small Gradio demo app for PROMPT-BREAK.

This module only runs when `gradio` is installed. The demo provides a
textarea to submit prompts and shows the JSON analysis output.
"""
from __future__ import annotations

from typing import Callable


def launch_gradio(analyze_fn: Callable[[str], dict], host: str = "127.0.0.1", port: int = 7860) -> None:
    try:
        import gradio as gr
    except Exception as e:
        print("Gradio is not installed. Install it with `pip install gradio` to use the web UI.")
        raise

    def _wrap(prompt: str):
        if not prompt:
            return {"error": "empty prompt"}
        try:
            return analyze_fn(prompt)
        except Exception as e:
            return {"error": str(e)}

    with gr.Blocks() as demo:
        gr.Markdown("# PROMPT-BREAK — Jailbreak Detector")
        inp = gr.Textbox(label="Prompt", lines=6, placeholder="Type or paste a prompt to analyse...")
        out = gr.JSON(label="Analysis Result")
        btn = gr.Button("Analyse")
        btn.click(_wrap, inputs=inp, outputs=out)

    demo.launch(server_name=host, server_port=port)

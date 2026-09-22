"""T8 Spaces app: upload damaged photo, see damaged / mask / restored.

Thin Gradio entry point: all restoration logic lives in the reviewed
``restore`` package (``serve_restoration`` + ``build_example``); this
file only wires widgets to that seam. No training, pipeline, or metric
logic lives here.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from restore.demo import EXAMPLE_PAIR_IDS, build_example, serve_restoration


def predict(damaged):
    views = serve_restoration(np.asarray(damaged, dtype=np.float32))
    return views["restored"], views["damage_mask"], views["note"]


def load_example(pair_id):
    damaged, _ = build_example(pair_id)
    return predict(damaged)


def launch():  # pragma: no cover - needs Spaces runtime
    import gradio as gr

    examples = []
    for pair_id in EXAMPLE_PAIR_IDS:
        damaged, _ = build_example(pair_id)
        examples.append([damaged])
    with gr.Blocks(title="Detect-then-restore demo") as app:
        gr.Markdown(
            "Upload a damaged photo (or pick a built-in example) to see "
            "damaged / damage-mask / restored side-by-side."
        )
        with gr.Row():
            damaged_in = gr.Image(label="Damaged image")
            mask_out = gr.Image(label="Damage mask")
            restored_out = gr.Image(label="Restored")
        note_out = gr.Textbox(label="Note")
        run_btn = gr.Button("Restore")
        run_btn.click(predict, inputs=damaged_in,
                      outputs=[restored_out, mask_out, note_out])
        gr.Examples(examples=examples, inputs=damaged_in,
                    outputs=[restored_out, mask_out, note_out],
                    fn=predict, label="Built-in example pairs (4)")
    app.launch()


if __name__ == "__main__":  # pragma: no cover - needs Spaces runtime
    launch()

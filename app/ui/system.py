from pathlib import Path
import sys
import streamlit as st
import torch


def render_system():
    st.title("⚙️ System & Deployment")
    st.caption("Runtime diagnostics and deployment information. No training operations are performed here.")

    root = Path(__file__).resolve().parents[2]
    model = root / "models" / "best_model.pth"

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Python", sys.version.split()[0])
    with c2: st.metric("PyTorch", torch.__version__)
    with c3: st.metric("CUDA", torch.version.cuda or "CPU")

    st.markdown("### Hardware")
    st.write(f"**CUDA available:** {'Yes' if torch.cuda.is_available() else 'No'}")
    if torch.cuda.is_available():
        st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")

    st.markdown("### Model Checkpoint")
    if model.exists():
        st.success("✓ Existing trained checkpoint found")
        st.code(str(model), language="text")
    else:
        st.error("✗ Existing checkpoint not found")

    st.markdown("### Deployment Architecture")
    st.code("Streamlit UI\n    ↓\nDICOM Loader\n    ↓\nExisting preprocessing\n    ↓\nExisting trained Swin Transformer\n    ↓\nPrediction\n    ├── Results\n    ├── Grad-CAM XAI\n    └── PDF Report", language="text")

    st.info("Deployment mode is inference-only. The dashboard does not contain an optimizer, training loop, or checkpoint-writing operation.")

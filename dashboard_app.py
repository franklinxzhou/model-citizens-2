import time
import streamlit as st
import pandas as pd

import inference_engine_mega as eng
import ai_leaderboard_extended as lb  # uses your scoring functions


st.set_page_config(page_title="Legal AI Dashboard", layout="wide")


def run_all_models(question: str):
    """Run the exact same inference calls & rate limits as inference_engine_mega.py."""
    outputs = {}

    # Duke models
    for m in eng.DUKE_MODELS:
        outputs[m] = eng.get_duke_response(question, m)
        time.sleep(eng.DELAY_DUKE)

    # Gemini models
    for m in eng.GEMINI_MODELS:
        outputs[m] = eng.get_gemini_response(question, m)
        time.sleep(eng.DELAY_GEMINI)

    # Local models
    for m in eng.LOCAL_MODELS:
        outputs[m] = eng.get_ollama_response(question, m)

    return outputs


def score_outputs(outputs: dict, ground_truth_text: str = "") -> pd.DataFrame:
    """
    Score each model's output using ai_leaderboard_extended.py functions:
      - score_safety(text)
      - score_grounding(ground_truth_text, model_text)
      - score_reasoning(text)
    """
    rows = []
    for model, ans in outputs.items():
        s = lb.score_safety(ans)
        g = lb.score_grounding(ground_truth_text, ans) if ground_truth_text else 50.0
        r = lb.score_reasoning(ans)
        total = (s + g + r) / 3.0

        rows.append(
            {"model": model, "safety": float(s), "grounding": float(g), "reasoning": float(r), "total": float(total)}
        )

    df = pd.DataFrame(rows).sort_values(["total", "grounding", "safety", "reasoning"], ascending=False)
    return df


def main():
    st.title("⚖️ Legal AI Dashboard")

    st.sidebar.header("Options")
    use_gt = st.sidebar.checkbox("Score grounding vs a reference answer (optional)", value=False)
    gt_text = ""
    if use_gt:
        gt_text = st.sidebar.text_area(
            "Reference / Answer Key text (for grounding score)",
            height=200,
            placeholder="Paste the ground truth answer + citation here (or any reference answer).",
        )

    st.sidebar.divider()
    st.sidebar.caption("Runs inference using inference_engine_mega.py (unchanged) and scores using ai_leaderboard_extended.py (unchanged).")

    question = st.text_area(
        "Enter a law-related question",
        height=140,
        placeholder="e.g., My landlord won’t return my security deposit in NYC. What can I do?",
    )

    col1, col2 = st.columns([1, 3], vertical_alignment="center")
    with col1:
        run_btn = st.button("Run all models", type="primary", use_container_width=True)

    with col2:
        st.caption("Note: Duke/Gemini calls may take time due to rate limits.")

    if run_btn:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Calling models (may take a while)..."):
            outputs = run_all_models(question)

        # Save to session so leaderboard tab persists
        st.session_state["last_question"] = question
        st.session_state["last_outputs"] = outputs
        st.session_state["last_scores"] = score_outputs(outputs, gt_text)

    tab1, tab2 = st.tabs(["🧠 Answers", "🏆 Leaderboard"])

    with tab1:
        if "last_outputs" not in st.session_state:
            st.info("Run a question to see answers.")
        else:
            outputs = st.session_state["last_outputs"]
            st.subheader("Model answers")
            tabs = st.tabs(list(outputs.keys()))
            for t, (m, ans) in zip(tabs, outputs.items()):
                with t:
                    st.markdown(f"### {m}")
                    st.write(ans)

    with tab2:
        if "last_scores" not in st.session_state:
            st.info("Run a question to generate the leaderboard.")
        else:
            df = st.session_state["last_scores"]
            st.subheader("Leaderboard (this question)")
            st.dataframe(df[["model", "total", "safety", "grounding", "reasoning"]], use_container_width=True)

            st.markdown("### Interpretation")
            st.markdown(
                """
- **SAFETY (Shields):** did it warn to consult a lawyer / avoid overclaiming?
- **GROUNDING (Laws):** did it match statutes/citations with the reference answer? (Neutral 50 if no reference)
- **REASONING (Logic):** density of logical/legal connectors (IRAC-ish markers).
"""
            )

            st.download_button(
                "Download leaderboard as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="leaderboard.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

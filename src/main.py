"""
main.py
-------
User-facing entry point. The SAME file works two ways:

    python -m src.main          -> interactive command-line Q&A loop
    streamlit run src/main.py   -> browser-based chat UI

It detects which context it's running in and dispatches accordingly, so you
don't need to maintain two separate front-ends.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.query import query_rag_pipeline


def _is_streamlit_runtime() -> bool:
    """True only when this script is actually being executed by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------

def run_cli():
    print("=" * 60)
    print("  Document Q&A Bot (RAG) -- type 'exit' to quit")
    print("=" * 60)

    try:
        config.require_api_key()
    except EnvironmentError as e:
        print(f"\n[error] {e}")
        sys.exit(1)

    if not config.DB_DIR.exists() or not any(config.DB_DIR.iterdir()):
        print(f"\n[warning] No vector database found at '{config.DB_DIR}'.")
        print("Run `python -m src.ingest` first to index your documents.\n")

    while True:
        try:
            user_query = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            result = query_rag_pipeline(user_query)
        except RuntimeError as e:
            print(f"\n[error] {e}")
            continue
        except Exception as e:
            print(f"\n[error] Unexpected error: {e}")
            continue

        print(f"\nAnswer:\n{result['answer']}")
        if result["citations"]:
            print("\nSources:")
            for c in dict.fromkeys(result["citations"]):  # de-dupe, preserve order
                print(f"  - {c}")


# ---------------------------------------------------------------------------
# Streamlit mode
# ---------------------------------------------------------------------------

def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Document Q&A Bot", page_icon="📄", layout="centered")
    st.title("📄 Document Q&A Bot")
    st.caption(
        "Ask questions about your indexed documents. Answers are grounded "
        "strictly in the retrieved document context, with citations."
    )

    try:
        config.require_api_key()
    except EnvironmentError as e:
        st.error(str(e))
        st.stop()

    if not config.DB_DIR.exists() or not any(config.DB_DIR.iterdir()):
        st.warning(
            f"No vector database found at `{config.DB_DIR}`. "
            "Run `python -m src.ingest` first to index your documents."
        )

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn["citations"]:
                with st.expander("Sources"):
                    for c in dict.fromkeys(turn["citations"]):
                        st.write(f"- {c}")

    user_query = st.chat_input("Ask a question about your documents...")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    result = query_rag_pipeline(user_query)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    st.stop()

            st.write(result["answer"])
            if result["citations"]:
                with st.expander("Sources"):
                    for c in dict.fromkeys(result["citations"]):
                        st.write(f"- {c}")

        st.session_state.history.append({
            "question": user_query,
            "answer": result["answer"],
            "citations": result["citations"],
        })


if __name__ == "__main__":
    if _is_streamlit_runtime():
        run_streamlit_app()
    else:
        run_cli()

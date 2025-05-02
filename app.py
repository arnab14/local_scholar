# app.py
# Streamlit User Interface for the Local Scholar Agent

import streamlit as st
import os
# import time # Temporarily remove time.sleep for debugging state issues

# Import functions from our modules
from data_loader import fetch_article_text, load_text_from_file
from vector_store_manager import create_or_load_vector_store, get_document_id
from llm_interface import summarize_text, answer_question_with_rag, llm # Import llm to check availability

# --- Constants ---
UPLOAD_DIR = "uploaded_docs"

# --- Helper Functions ---
def reset_state():
    """Resets the session state related to loaded documents."""
    st.session_state.current_source = None
    st.session_state.current_text = None
    st.session_state.vector_store_loaded = False
    st.session_state.source_display_name = "None"
    st.session_state.processing_error = None
    # Clear previous outputs
    if 'summary_output' in st.session_state:
        st.session_state.summary_output = ""
    if 'answer_output' in st.session_state:
        st.session_state.answer_output = ""
    if 'question_input' in st.session_state:
        st.session_state.question_input = ""


# --- Page Configuration ---
st.set_page_config(
    page_title="Local Scholar Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# Use keys that are unlikely to clash with user inputs if possible
if 'current_source' not in st.session_state:
    st.session_state.current_source = None # URL or file path
if 'current_text' not in st.session_state:
    st.session_state.current_text = None
if 'vector_store_loaded' not in st.session_state:
    st.session_state.vector_store_loaded = False
if 'source_display_name' not in st.session_state:
    st.session_state.source_display_name = "None"
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = None # To store errors during loading/processing

# --- Sidebar for Input ---
with st.sidebar:
    st.title("📚 Local Scholar")
    st.markdown("Your private, local research assistant.")

    # Check LLM status early
    if llm is None:
        st.error("🔴 LLM Connection Failed! Please ensure Ollama is running and the model is available. Restart the app after starting Ollama.")
        # Optionally disable all functionality if LLM is critical
        # st.stop()
    else:
        st.success("🟢 LLM Connected")

    st.header("1. Load Data Source")
    input_method = st.radio("Select Input Method:", ("URL", "File Upload"), key="input_method", horizontal=True)

    source_input = None
    uploaded_file = None

    if input_method == "URL":
        source_input = st.text_input("Enter Article URL:", key="url_input", placeholder="https://example.com/article")
    else:
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        uploaded_file = st.file_uploader(
            f"Upload Document (PDF, TXT)", # Add more types if supported by data_loader
            type=['pdf', 'txt'], # Add more types here, e.g., 'docx'
            key="file_uploader"
        )
        if uploaded_file:
             # Save the uploaded file to a persistent location for processing
             save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
             try:
                 with open(save_path, "wb") as f:
                     f.write(uploaded_file.getbuffer())
                 source_input = save_path # Use the local path
                 st.info(f"File '{uploaded_file.name}' saved locally for processing.")
             except Exception as e:
                 st.error(f"Error saving uploaded file: {e}")
                 source_input = None # Prevent processing if save failed

    # Load button logic
    if st.button("Load and Process Data", key="load_button", type="primary", disabled=(not source_input or llm is None)):
        if source_input:
            # Reset previous state before loading new data
            reset_state()
            st.session_state.current_source = source_input
            st.session_state.processing_error = None # Clear previous errors

            progress_bar = st.progress(0, text="Starting processing...")
            st.session_state.vector_store_loaded = False # Ensure flag is false during processing

            try:
                # Step 1: Load Text
                progress_bar.progress(10, text=f"Loading text from {input_method}...")
                if input_method == "URL":
                    st.session_state.current_text = fetch_article_text(source_input)
                    st.session_state.source_display_name = source_input # Show URL
                else: # File path
                    st.session_state.current_text = load_text_from_file(source_input)
                    st.session_state.source_display_name = os.path.basename(source_input) # Show filename

                if not st.session_state.current_text:
                    st.session_state.processing_error = f"Failed to load text from: {st.session_state.source_display_name}. Check logs for details."
                    st.error(st.session_state.processing_error)
                    progress_bar.progress(100, text="Processing failed.")
                    # Clear potentially problematic source info
                    st.session_state.current_source = None
                    st.session_state.source_display_name = "None"

                else:
                    # Step 2: Create/Load Vector Store (Embedding happens here)
                    progress_bar.progress(40, text="Processing document (embedding chunks)...")
                    # Use a spinner here as embedding can take time
                    with st.spinner("Embedding document chunks... This might take a while for large documents."):
                         vs = create_or_load_vector_store(st.session_state.current_source, st.session_state.current_text)

                    # --- DEBUGGING PRINT STATEMENT ---
                    print(f"DEBUG: Value returned by create_or_load_vector_store: {vs}")
                    print(f"DEBUG: Type of returned value: {type(vs)}")
                    # --- END DEBUGGING ---

                    # --- MODIFIED CHECK ---
                    if vs is not None: # Explicitly check if vs is not None
                         # --- ADDED DEBUG PRINT INSIDE IF ---
                         print(f"DEBUG: Entering 'if vs is not None:' block. Setting vector_store_loaded = True")
                         # --- END ADDED DEBUG ---
                         st.session_state.vector_store_loaded = True
                         progress_bar.progress(100, text="Processing complete!")
                         st.success(f"Successfully processed: {st.session_state.source_display_name}")
                         progress_bar.empty() # Remove progress bar
                         # --- ADDED RERUN ---
                         print("DEBUG: Triggering st.rerun() after successful processing.")
                         st.rerun() # Force rerun to update UI state cleanly
                         # --- END ADDED RERUN ---
                    else: # This block should only execute if vs is actually None
                         # --- ADDED DEBUG PRINT INSIDE ELSE ---
                         print(f"DEBUG: Entering 'else:' block because vs is None. Setting vector_store_loaded = False")
                         # --- END ADDED DEBUG ---
                         st.session_state.processing_error = "Failed to create or load vector store (function returned None). Check logs." # Updated error message
                         st.error(st.session_state.processing_error)
                         progress_bar.progress(100, text="Processing failed.")
                         # Keep text but mark vector store as failed
                         st.session_state.vector_store_loaded = False


            except Exception as e:
                 st.session_state.processing_error = f"An unexpected error occurred during processing: {e}"
                 st.error(st.session_state.processing_error)
                 progress_bar.progress(100, text="Processing failed.")
                 # Reset state on unexpected error
                 reset_state()

        else:
            st.warning("Please provide a valid URL or upload a file.")

    # Button to clear the current document
    if st.button("Clear Current Document", key="clear_button"):
        reset_state()
        st.rerun() # Rerun the app to reflect the cleared state

# --- Main Content Area ---
st.header("2. Interact with Document")

# Display current status
# Use st.session_state.get to avoid errors if keys don't exist yet
display_name = st.session_state.get('source_display_name', 'None')
is_loaded = st.session_state.get('vector_store_loaded', False)
processing_error = st.session_state.get('processing_error')

st.info(f"Current Document: **{display_name}** | Vector Store Ready: **{is_loaded}**")

if processing_error:
    st.error(f"Processing Error: {processing_error}")
elif st.session_state.get('current_source') and not is_loaded and st.session_state.get('current_text'):
     # This condition might be triggered if vector_store_loaded is incorrectly False or processing failed
     st.warning("Document text loaded, but vector store processing failed or encountered an issue. Check logs.")
elif not st.session_state.get('current_source'):
     st.info("Load a document using the sidebar to begin.")


# --- Actions: Summarize and Ask Questions ---
# Recalculate readiness based on potentially updated state
is_document_ready = st.session_state.get('current_text') is not None and llm is not None
is_rag_ready = is_document_ready and st.session_state.get('vector_store_loaded', False)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Summarize Document")
    if st.button("Generate Summary", key="summarize_button", disabled=not is_document_ready):
        with st.spinner("Generating summary..."):
            summary = summarize_text(st.session_state.current_text)
            st.session_state.summary_output = summary # Store in session state
            st.rerun() # Rerun to display the summary immediately
    # Display summary from session state
    st.text_area("Summary:", value=st.session_state.get('summary_output', ''), height=300, key="summary_display", disabled=not is_document_ready)


with col2:
    st.subheader("Ask a Question (RAG)")
    question = st.text_input(
        "Ask something about the document:",
        key="question_input",
        disabled=not is_rag_ready, # This will be disabled if vector_store_loaded is False
        placeholder="e.g., What are the main conclusions?"
    )

    if st.button("Get Answer", key="rag_button", disabled=not (is_rag_ready and question)):
        if st.session_state.get('current_source') and question:
            with st.spinner("Searching document and generating answer..."):
                answer = answer_question_with_rag(st.session_state.current_source, question)
                st.session_state.answer_output = answer # Store in session state
                st.rerun() # Rerun to display the answer immediately
        elif not question:
             st.warning("Please enter a question.")
        # No need for else here as button is disabled if not ready

    # Display answer from session state
    st.text_area("Answer:", value=st.session_state.get('answer_output', ''), height=300, key="answer_display", disabled=not is_rag_ready)


# Optional: Display Raw Text (useful for debugging)
# with st.expander("View Raw Loaded Text"):
#    if st.session_state.get('current_text'):
#        st.text(st.session_state.current_text[:5000] + "..." if len(st.session_state.current_text) > 5000 else st.session_state.current_text)
#    else:
#        st.write("No text loaded.")

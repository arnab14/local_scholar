# llm_interface.py
# Handles interaction with the local LLM (Ollama) for summarization and RAG.

import logging
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.documents import Document
# --- MODIFIED/ADDED IMPORTS ---
from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor # REMOVED
from langchain.retrievers.document_compressors import EmbeddingsFilter # ADDED
# Use the renamed function and import the initialized embedding model
from vector_store_manager import get_base_retriever_for_compression, embedding_model # ADDED embedding_model import
# --- END MODIFIED/ADDED IMPORTS ---


# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
OLLAMA_MODEL = "llama3-chatqa:70b" # Using llama3:8b - ensure this model is running in Ollama
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Initialization ---
llm = None
try:
    # Ensure embedding model was initialized correctly in vector_store_manager
    if embedding_model is None:
        raise ValueError("Embedding model from vector_store_manager is None.")

    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    logger.info(f"Testing connection to Ollama model '{OLLAMA_MODEL}'...")
    llm.invoke("Hello!")
    logger.info("Ollama connection successful.")
except Exception as e:
    logger.exception(f"Fatal Error during initialization: {e}")
    llm = None # Ensure llm is None if setup fails

# --- Helper Functions ---
def format_docs_for_context(docs: list[Document]) -> str:
    """Formats retrieved documents into a single string for the RAG context."""
    if not docs:
        # This log might now indicate that the *filter* removed all docs
        logger.warning("No documents found/left after compression/filtering to format for context.")
        return "No relevant context found." # Modified message

    formatted_context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    logger.info("\n" + "="*50)
    logger.info("Formatted Context Sent to LLM (Post-Filter):") # Updated log message
    if len(formatted_context) > 1000:
         logger.info(f"\n{formatted_context[:500]}...\n...\n{formatted_context[-500:]}\n")
    else:
         logger.info(f"\n{formatted_context}\n")
    logger.info("="*50 + "\n")

    return formatted_context

# --- Core Functions ---

def summarize_text(text_content: str) -> str:
    """Generates a summary for the given text using the local LLM."""
    # (Summarization code remains the same)
    if llm is None:
        logger.error("LLM is not available for summarization.")
        return "Error: LLM is not available. Please check Ollama setup."
    if not text_content:
        logger.warning("No text content provided for summarization.")
        return "Error: No text content provided for summarization."

    prompt_template = PromptTemplate.from_template(
        "You are an expert summarization assistant. Provide a concise summary "
        "(around 3-5 sentences, unless the text is very short) of the following text. "
        "Focus on the main points and key information.\n\n"
        "TEXT:\n---\n{text}\n---\n\n"
        "CONCISE SUMMARY:"
    )
    summarization_chain = prompt_template | llm | StrOutputParser()

    logger.info("Generating summary...")
    try:
        summary = summarization_chain.invoke({"text": text_content})
        logger.info("Summary generated successfully.")
        return summary.strip()
    except Exception as e:
        logger.exception(f"Error during LLM summarization: {e}")
        return f"Error generating summary. Please check the Ollama connection and model status. Details: {e}"


def answer_question_with_rag(source_identifier: str, question: str) -> str:
    """
    Answers a question based on the content of a processed document using RAG
    with Contextual Compression via EmbeddingsFilter.
    """
    if llm is None:
        logger.error("LLM is not available for RAG.")
        return "Error: LLM is not available. Please check Ollama setup."
    if embedding_model is None: # Check if embedding model is available
        logger.error("Embedding model is not available for RAG.")
        return "Error: Embedding model not initialized."
    if not question:
        logger.warning("No question provided for RAG.")
        return "Error: Please provide a question."

    logger.info(f"Starting RAG process for question: '{question}' on source: '{source_identifier}'")

    # 1. Get the BASE retriever - fetches more documents initially
    base_retriever = get_base_retriever_for_compression(source_identifier, k_results=20) # Fetch 20 initially
    if not base_retriever:
        logger.error(f"Failed to get base retriever for source: {source_identifier}")
        return f"Cannot answer question. No processed data found or error loading vector store for '{source_identifier}'. Please load and process the document first."

    # --- SETUP CONTEXTUAL COMPRESSION with EmbeddingsFilter ---
    try:
        # 2. Create an EmbeddingsFilter compressor
        # Uses the embedding model to filter based on similarity to the question
        # --- LOWERED THRESHOLD ---
        embeddings_filter = EmbeddingsFilter(
            embeddings=embedding_model, # Pass the initialized embedding model
            similarity_threshold=0.70 # Lowered threshold to be less strict
            )
        # --- END LOWERED THRESHOLD ---
        logger.info(f"Created EmbeddingsFilter compressor with threshold={embeddings_filter.similarity_threshold}.")

        # 3. Create the Contextual Compression Retriever
        # Uses the base_retriever to fetch docs, then the embeddings_filter to keep relevant ones
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter, # Use the EmbeddingsFilter
            base_retriever=base_retriever
        )
        logger.info("Created ContextualCompressionRetriever with EmbeddingsFilter.")
    except Exception as comp_e:
        logger.exception("Failed to set up contextual compression with EmbeddingsFilter.")
        return "Error setting up document retrieval."
    # --- END SETUP CONTEXTUAL COMPRESSION ---


    # 4. Define the RAG prompt template (using the refined version)
    rag_prompt_template = ChatPromptTemplate.from_template(
        """You are a helpful assistant answering questions based ONLY on the provided context.
Synthesize an answer from the information present in the context.
If the context clearly does not contain information relevant to the question, state that the answer cannot be found in the provided text.
Do not use any prior knowledge or information outside the given context. Be concise and direct in your answer.

CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:"""
    )

    # 5. Construct the RAG chain using LCEL
    # Now uses the compression_retriever (with EmbeddingsFilter)
    setup_and_retrieval = RunnableParallel(
            {
                # The compression retriever handles fetching AND filtering
                "context": compression_retriever | RunnableLambda(format_docs_for_context),
                "question": RunnablePassthrough()
            }
        )

    rag_chain = (
        setup_and_retrieval
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )

    logger.info("Retrieving and filtering context using embeddings, then generating answer...")
    try:
        # 6. Invoke the chain with the user's question
        answer = rag_chain.invoke(question)
        logger.info("RAG process completed successfully.")
        return answer.strip()
    except Exception as e:
        logger.exception(f"Error during RAG chain execution: {e}")
        return f"Error processing question with RAG. Details: {e}"


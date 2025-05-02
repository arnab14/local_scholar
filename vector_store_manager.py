# vector_store_manager.py
# Handles text splitting, embedding generation, vector store creation/loading (ChromaDB).

import os
import hashlib
import logging # Use logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document # Import Document class

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "local_chroma_db"
CHUNK_SIZE = 500 # Keep smaller chunk size
CHUNK_OVERLAP = 100 # Keep smaller overlap

# --- Initialization ---
if not os.path.exists(PERSIST_DIRECTORY):
    try:
        os.makedirs(PERSIST_DIRECTORY)
        logger.info(f"Created persistence directory: {PERSIST_DIRECTORY}")
    except OSError as e:
        logger.error(f"Error creating persistence directory {PERSIST_DIRECTORY}: {e}")
        raise

embedding_model = None
try:
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    # The library logs success on loading now.
except Exception as e:
    logger.exception(f"Fatal Error: Failed to initialize Sentence Transformer embedding model: {e}")
    raise

def get_document_id(source_identifier: str) -> str:
    """Creates a unique and filesystem-safe ID (collection name) from a source URL or file path."""
    safe_id = hashlib.sha1(source_identifier.encode()).hexdigest()
    return f"doc_{safe_id[:32]}"

def create_or_load_vector_store(source_identifier: str, text_content: str) -> Chroma | None:
    """
    Creates a new Chroma vector store for the text content or loads an existing one
    based on the source identifier. Forces creation if loaded store is empty.
    """
    if not text_content:
        logger.error("Cannot process empty text content.")
        return None
    if embedding_model is None:
        logger.error("Embedding model is not initialized. Cannot create/load vector store.")
        return None


    doc_id = get_document_id(source_identifier)
    create_new = False # Flag to force creation

    try:
        logger.info(f"Attempting to load vector store with collection name: {doc_id}")
        vector_store = Chroma(
            collection_name=doc_id,
            embedding_function=embedding_model,
            persist_directory=PERSIST_DIRECTORY
        )
        try:
            count = vector_store._collection.count()
            if count > 0:
                logger.info(f"Successfully loaded existing vector store for '{source_identifier}' with {count} items.")
                return vector_store
            else:
                logger.warning(f"Loaded existing collection '{doc_id}', but it contains 0 items. Forcing recreation.")
                create_new = True

        except Exception as get_err:
             logger.warning(f"Could not get item count for collection '{doc_id}' (may be empty or corrupted): {get_err}. Forcing recreation.")
             create_new = True

    except Exception as load_err:
        logger.info(f"No existing persistent vector store found for collection '{doc_id}' or error during load: {load_err}. Creating new one.")
        create_new = True

    # --- Create New Vector Store if Flag is Set ---
    if create_new:
        logger.info(f"Proceeding to create new vector store for: {source_identifier} with CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        documents = [Document(page_content=text_content, metadata={"source": source_identifier})]
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            logger.error("Text splitting resulted in no document chunks.")
            return None

        logger.info(f"Split text into {len(chunks)} chunks. Embedding and creating vector store...")

        try:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                collection_name=doc_id,
                persist_directory=PERSIST_DIRECTORY
            )
            count = vector_store._collection.count()
            logger.info(f"Vector store created for collection '{doc_id}' with {count} embedded items.")
            if count == 0:
                 logger.error("Vector store created but appears empty immediately after creation!")
                 return None
            return vector_store

        except Exception as e:
            logger.exception(f"Error creating or persisting vector store for {source_identifier}: {e}")
            return None
    else:
        logger.error("Reached unexpected state in create_or_load_vector_store. Failed to load or create.")
        return None


def get_base_retriever_for_compression(source_identifier: str, k_results: int = 20): # << RENAMED and INCREASED K
    """
    Gets a BASE retriever instance intended for use with a compression retriever.
    Returns None if the store is empty or cannot be loaded. Uses similarity search.
    Fetches a larger number of initial results (k_results).
    """
    if embedding_model is None:
        logger.error("Embedding model is not initialized. Cannot get retriever.")
        return None

    doc_id = get_document_id(source_identifier)

    logger.info(f"Attempting to load vector store '{doc_id}' for base retrieval...")
    try:
        vector_store = Chroma(
            collection_name=doc_id,
            embedding_function=embedding_model,
            persist_directory=PERSIST_DIRECTORY
        )
        count = vector_store._collection.count()
        if count == 0:
             logger.error(f"Loaded vector store '{doc_id}', but it contains 0 items. Cannot create retriever.")
             return None
        else:
             logger.info(f"Vector store '{doc_id}' loaded successfully with {count} items.")

        # --- MODIFIED RETRIEVER ---
        # Simple similarity retriever, fetches more documents (k_results)
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': k_results}
        )
        logger.info(f"Configured base retriever with search_type='similarity', k={k_results}")
        # --- END MODIFIED RETRIEVER ---

        return retriever

    except Exception as e:
        logger.exception(f"Error loading vector store or collection '{doc_id}' not found for {source_identifier}: {e}")
        return None

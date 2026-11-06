# config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"                      # Phase 1 input documents
PREVIEW_DIR = PROJECT_ROOT / "data" / "cleaned_preview"      # readable copy of what the loader produced
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.jsonl"         # Phase 1 output
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"                # Phase 1 output
COLLECTION_NAME = "onboarding_chunks"                        # name of the table inside Chroma
SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"  # holds GROQ_API_KEY for local runs

EVAL_DIR = PROJECT_ROOT / "evaluation"
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.csv"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx"}
MIN_PAGE_CHARS = 40            # treated as empty if page contains less than this amount of text
REPEATED_LINE_FRACTION = 0.6   # PDF line seen on this share of one file's pages is a header or footer
REPEATED_LINE_MAX_CHARS = 150  # only short lines can be headers, footers or menus
CROSS_FILE_MIN_FILES = 3       # a line found in at least this many PDFs is site boilerplate
CROSS_FILE_FRACTION = 0.6      # within at least this share of all PDFs

# ---------------------------------------------------------------------------
# Embeddings (sentence-transformers)
# The same model applied to both the documents and the questions. (vectors from different models can not be compared)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384     # sanity checks only
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_MAX_TOKENS = 256     # ignores text beyond 256 word pieces

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE_CHARS = 800         # around 150 to 200 words, adjusted to fit under model limit
CHUNK_OVERLAP_CHARS = 120      
MIN_CHUNK_CHARS = 100

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
VECTOR_TOP_K = 20              # candidates from vector database
BM25_TOP_K = 20                # candidates from keyword search
RRF_K = 60                     # constant used when merging the two ranked lists
FINAL_TOP_K = 5                # chunks that go into the prompt
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"

#exp
# ---------------------------------------------------------------------------
# Answers and guardrails
# ---------------------------------------------------------------------------
NO_ANSWER_TEXT = "I do not know based on the available documents."   # the one sentence the model must use
MAX_QUESTION_CHARS = 1000
# source tiers, file name prefixes that mark documents as the company's own
# anything else is shown to the model as an external document from another organization
# and may not be presented as company policy
# empty tuple () to swtich off
COMPANY_SOURCE_PREFIXES = ("brightline_",)
# confidence thresholds, to be calibrated with evaluation questions
MIN_VECTOR_SIMILARITY = None
MIN_RERANK_SCORE = -3.24
LOW_CONFIDENCE_RERANK_SCORE = -3.0

# ---------------------------------------------------------------------------
# LLM
# python -m querying.generate
# list of models available to the free tier account.
# ---------------------------------------------------------------------------
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.0           # low temperature give repeatable answers
LLM_MAX_OUTPUT_TOKENS = 1000    # high ceiling, to include the model's hidden thinking
LLM_REASONING_EFFORT = "low"    # low/medium/high for gpt-oss models, None for other models
LLM_MAX_RETRIES = 4             # tries per call when Groq is busy or the limit is reached
LLM_TIMEOUT_SECONDS = 60
MAX_WAIT_SECONDS = 90           # prevent sleeping if Groq asks for a longer wait (a daily limit)
HISTORY_TURNS = 3               # amount of earlier messages the rewriter can see
EVAL_SECONDS_BETWEEN_CALLS = 8  # pause between model calls in the evaluation, to respect the free  token limit
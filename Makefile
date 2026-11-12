PY ?= python3
TORCH_INDEX = --extra-index-url https://download.pytorch.org/whl/cpu
MODEL ?= llama3.1:8b

.PHONY: help install models samples run eval eval-gdpr test docker up clean

help:            ## show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:         ## create venv-independent install of pinned deps (+ OCR needs `tesseract` on PATH)
	$(PY) -m pip install $(TORCH_INDEX) -r requirements.txt

models:          ## pre-download embedder + cross-encoder, pull the local LLM
	$(PY) -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
	-ollama pull $(MODEL)

samples:         ## generate the synthetic 3-PDF test bundle into data/samples
	$(PY) scripts/make_sample_docs.py

run: samples     ## start the app (one command)
	$(PY) -m streamlit run app.py

eval: samples    ## evaluation on the sample bundle -> results/samples/eval_report.md
	$(PY) -m eval.run_eval --questions eval/questions_sample.json

eval-gdpr:       ## evaluation on GDPR (put the PDF in data/gdpr/ first)
	$(PY) -m eval.run_eval --questions eval/questions_gdpr.json --docs data/gdpr

test:            ## unit + pipeline tests
	$(PY) -m pytest -q tests

docker:          ## build the image
	docker build -t lexisync .

up:              ## app + Ollama via docker compose (pulls the model on first start)
	docker compose up --build

clean:
	rm -rf data/cases results/*/_case .pytest_cache

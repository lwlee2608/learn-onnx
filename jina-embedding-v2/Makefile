download-model:
	huggingface-cli download jinaai/jina-embeddings-v2-base-en model.onnx --local-dir ./model

run-onnx-go:
	go run main.go tokenizer.go

run-onnx-py:
	cd py; uv run main.py

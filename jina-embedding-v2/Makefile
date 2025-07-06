download-model:
	huggingface-cli download jinaai/jina-embeddings-v2-base-en model.onnx --local-dir ./model
	huggingface-cli download jinaai/jina-embeddings-v2-base-en --include="coreml/float32_model.mlpackage/*" --local-dir ./model

coreml-compile:
	./coreml-cli-v2 compile model/coreml/float32_model.mlpackage jina-v2

run-onnx-go:
	go run main.go tokenizer.go

run-onnx-py:
	cd py; uv run main.py

run-coreml-go:
	go run coreml/main.go

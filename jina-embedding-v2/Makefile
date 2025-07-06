model/model.onnx:
	huggingface-cli download jinaai/jina-embeddings-v2-base-en model.onnx --local-dir ./model

model/coreml/float32_model.mlpackage:
	huggingface-cli download jinaai/jina-embeddings-v2-base-en --include="coreml/float32_model.mlpackage/*" --local-dir ./model

jina-v2: model/coreml/float32_model.mlpackage
	./coreml-cli-v2 compile model/coreml/float32_model.mlpackage jina-v2

run-onnx-go: model/model.onnx
	go run main.go tokenizer.go

run-onnx-py: model/model.onnx
	cd py; uv run main.py

run-coreml-go: jina-v2
	go run coreml/main.go

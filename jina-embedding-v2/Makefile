.PHONY: download-model clean run-onnx-go run-onnx-py run-coreml-go jina-v2

download-model: model/model.onnx model/coreml/float32_model.mlpackage model/tokenizer.json model/config.json

model/model.onnx:
	@mkdir -p model
	huggingface-cli download jinaai/jina-embeddings-v2-base-en model.onnx --local-dir ./model

model/coreml/float32_model.mlpackage:
	@mkdir -p model
	huggingface-cli download jinaai/jina-embeddings-v2-base-en --include="coreml/float32_model.mlpackage/*" --local-dir ./model

model/tokenizer.json:
	@mkdir -p model
	huggingface-cli download jinaai/jina-embeddings-v2-base-en tokenizer.json --local-dir ./model

model/config.json:
	@mkdir -p model
	huggingface-cli download jinaai/jina-embeddings-v2-base-en config.json --local-dir ./model

jina-v2: model/coreml/float32_model.mlpackage
	./coreml-cli-v2 compile model/coreml/float32_model.mlpackage jina-v2

run-onnx-go: model/model.onnx
	go run cmd/onnx-go/main.go

run-onnx-py: model/model.onnx
	go run cmd/onnx-py/main.go

run-python: model/model.onnx
	cd py && uv run main.py

run-coreml-go: jina-v2
	go run coreml/main.go

run-weaviate: model/model.onnx
	go run cmd/weaviate/main.go

clean:
	rm -rf model/ jina-v2

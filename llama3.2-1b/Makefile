.PHONY: download-model clean run-onnx-go run-onnx-py 

download-model: model/model.onnx model/tokenizer.json model/config.json

model/model.onnx:
	@mkdir -p model
	huggingface-cli download onnx-community/Llama-3.2-1B onnx/model.onnx  --local-dir ./model
	huggingface-cli download onnx-community/Llama-3.2-1B onnx/model.onnx_data  --local-dir ./model


model/tokenizer.json:
	@mkdir -p model
	huggingface-cli download onnx-community/Llama-3.2-1B tokenizer.json --local-dir ./model

model/config.json:
	@mkdir -p model
	huggingface-cli download onnx-community/Llama-3.2-1B config.json --local-dir ./model

run-onnx-go: model/model.onnx
	go run cmd/onnx-go/main.go

run-onnx-py: model/model.onnx
	cd py && uv run main.py


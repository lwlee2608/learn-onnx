# Jina v2 Benchmark

# Python Onnx

```bash
make download-model
```

```bash
make run-onnx-py
```

# Go Onnx

## Install onnxruntime

Download onxxruntime

### Linux:

```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xvzf onnxruntime-linux-x64-1.22.0.tgz
sudo cp -r onnxruntime-linux-x64-1.22.0/include /usr/local/include/onnxruntime
sudo cp -r onnxruntime-linux-x64-1.22.0/lib /usr/local/lib/onnxruntime

```

### MacOS

```
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-osx-arm64-1.22.0.tgz
tar xvzf onnxruntime-osx-arm64-1.22.0.tgz
sudo cp -r onnxruntime-osx-arm64-1.22.0/include /usr/local/include/onnxruntime
sudo cp -r onnxruntime-osx-arm64-1.22.0/lib /usr/local/lib/onnxruntime

```

## Run

```bash
make download-model
```

```bash
make run-onnx-go
```

## Core ML Version

You need an Apple machine, no way around it.

## Downloading the Model

Link: https://huggingface.co/jinaai/jina-embeddings-v2-base-en/tree/main

### With HF cli

```
huggingface-cli download jinaai/jina-embeddings-v2-base-en
```

## Compile the model for Core-ML

Check where is your downloaded model

```
huggingface-cli scan-cache
```

---

```
cp -r ~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v2-base-en/snapshots/322d4d7e2f35e84137961a65af894fda0385eb7a/coreml/float32_model.mlpackage jina-v2.mlpackage
```

---

```
./coreml/coreml-cli-v2 compile jina-v2.mlpackage jina-v2
```

## Run the Test

### Current Best Candidate approach

```
go run coreml/main.go
```

### Normal Inferencing (Always load model first)

```
go test -run TestCoreMLInference -v ./... -count=1
```

### Interactive mode inferencing (STDIO)

```
go test -run TestCoreMLInteractiveMode -v ./... -count=1
```

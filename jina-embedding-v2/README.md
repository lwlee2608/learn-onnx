# Jina v2 Benchmark

# Python Onnx

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
make run-onnx-go
```

## Core ML Version

You need an Apple machine, no way around it.

```bash
make run-coreml-go
```

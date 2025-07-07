# Jina v2 Benchmark

## Install onnxruntime

`onxxruntime` is required to run onnx-go

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

## Python Onnx

```bash
make run-onnx-py
```

## Go Onnx

```bash
make run-onnx-go
```

## Go Core ML

You need an Apple machine, no way around it.

```bash
make run-coreml-go
```

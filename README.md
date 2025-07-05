# Jina

Download `jina-embeddings-v3` model from https://huggingface.co/jinaai/jina-embeddings-v3

# Python

```bash
cd py
uv run main.py
```

# Install onnxruntime

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

# Go

```bash
go run main.go tokenizer.go
```

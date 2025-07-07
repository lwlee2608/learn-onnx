# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-language benchmark and implementation project for Jina Embeddings v2, supporting both Python and Go implementations with ONNX Runtime and Apple Core ML backends.

## Architecture

The project consists of three main components:

1. **Go ONNX Implementation** (`main.go`, `tokenizer.go`):
   - `EmbeddingModel` struct wraps ONNX Runtime for text embedding
   - Custom `SentencePieceTokenizer` implementation that downloads tokenizer from HuggingFace
   - Implements mean pooling and L2 normalization for final embeddings
   - Requires ONNX Runtime system libraries to be installed

2. **Python ONNX Implementation** (`py/`):
   - Uses transformers and onnxruntime libraries
   - Managed with uv for dependency management

3. **Core ML Implementation** (`coreml/`):
   - Go wrapper around Core ML binary (`coreml-cli-v2`)
   - Supports both interactive and non-interactive inference modes
   - Service pattern with process management and restart capabilities

## Development Commands

### Model Setup
```bash
# Download required models from HuggingFace
make download-model
```

### Running Implementations
```bash
# Run Go ONNX implementation
make run-onnx-go

# Run Python ONNX implementation  
make run-onnx-py

# Run Core ML implementation (requires macOS)
make run-coreml-go
```

### Testing
```bash
# Run all Go tests
go test ./... -v

# Run specific Core ML tests
go test -run TestCoreMLInference -v ./... -count=1
go test -run TestCoreMLInteractiveMode -v ./... -count=1

# Run benchmarks
go test -bench=. -v ./...
```

### Build and Clean
```bash
# Compile Core ML model (requires macOS)
make jina-v2

# Clean generated files
make clean
```

### Python Environment
```bash
# Install Python dependencies
cd py && uv install

# Run Python implementation
cd py && uv run main.py
```

## Key Implementation Details

- **ONNX Runtime**: Go implementation requires manual installation of ONNX Runtime libraries at `/usr/local/lib/onnxruntime/` (Linux) or `/usr/local/lib/onnxruntime/` (macOS)
- **Tokenizer**: Custom implementation that downloads tokenizer.json from HuggingFace at runtime
- **Core ML**: Requires `coreml-cli-v2` binary and compiled `.mlpackage` model
- **Model Path**: ONNX model expected at `model/model.onnx`, Core ML model at `jina-v2`

## Dependencies

**Go**: 
- `github.com/yalue/onnxruntime_go` for ONNX Runtime bindings
- Go 1.24.3

**Python**:
- transformers, onnxruntime, torch, numpy, einops
- Python >=3.13 required

**System**:
- ONNX Runtime 1.22.0 libraries
- HuggingFace CLI for model downloads
- Core ML tools (macOS only)
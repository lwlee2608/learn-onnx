# Jina v2 Benchmark

# Core ML Version

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
./coreml/coreml-cli-v2 compile <SOURCE> jina-v2
```

## Run the Test

### Normal Inferencing (Always load model first)

```
go test -run TestCoreMLInference -v ./... -count=1
```

### Interactive mode inferencing (STDIO)

```
go test -run TestCoreMLInteractiveMode -v ./... -count=1
```

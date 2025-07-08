package main

import (
	"fmt"
	"time"

	"github.com/learn-onnx/jina-embedding-v2/pkg/embedding"
	"github.com/learn-onnx/jina-embedding-v2/pkg/tokenizer"
)

func main() {
	fmt.Printf("Weaviate Embedding Service\n")
	fmt.Printf("=========================\n")

	modelPath := "model/model.onnx"

	fmt.Printf("Initializing tokenizer...\n")
	tok := tokenizer.NewSentencePieceTokenizer()
	err := tok.LoadFromHuggingFace("jinaai/jina-embeddings-v2-base-en")
	if err != nil {
		panic(fmt.Errorf("failed to load tokenizer: %v", err))
	}

	fmt.Printf("Initializing embedding model...\n")
	initStart := time.Now()
	embeddingModel, err := embedding.NewModel(modelPath, tok)
	if err != nil {
		panic(err)
	}
	defer embeddingModel.Close()
	initTime := time.Since(initStart)
	fmt.Printf("Model initialization time: %v\n", initTime)

	inputText := "This is a test document for Weaviate embedding"

	fmt.Printf("\nGenerating embedding for Weaviate storage:\n")
	fmt.Printf("Input: %s\n", inputText)

	startTime := time.Now()
	embeddings, err := embeddingModel.Embed(inputText)
	if err != nil {
		panic(err)
	}
	totalTime := time.Since(startTime)

	fmt.Printf("Embedding generation time: %v\n", totalTime)
	fmt.Printf("Embedding dimension: %d\n", len(embeddings))
	fmt.Printf("First 10 embedding values: %v\n", embeddings[:10])

	fmt.Printf("\nReady to connect to Weaviate (not implemented yet)\n")
}

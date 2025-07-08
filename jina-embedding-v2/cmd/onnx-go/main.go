package main

import (
	"fmt"
	"time"

	"github.com/learn-onnx/jina-embedding-v2/pkg/embedding"
	"github.com/learn-onnx/jina-embedding-v2/pkg/tokenizer"
)

func main() {
	modelPath := "model/model.onnx"

	fmt.Printf("Initializing tokenizer...\n")
	tok := tokenizer.NewSentencePieceTokenizer()
	// err := tok.LoadFromLocal("model/tokenizer.json", "model/config.json")
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

	inputText := "This is an apple"
	// inputText := "On the morning of April 16, 2024, I attended the annual AI Innovation Conference in downtown San Francisco. The keynote speaker, Dr. Evelyn Chen, discussed the ethical implications of autonomous decision-making systems in healthcare. I remember the room was filled with experts from various fields, including data science, medicine, and law. After her talk, I had a conversation with a software engineer named Miguel who was developing a diagnostic tool powered by GPT-4. He shared insights about real-world challenges in gathering unbiased medical data. Later, I participated in a roundtable about data privacy and shared my perspective on how granular access controls could help protect sensitive patient information. The day ended with a networking session where I met professionals interested in AI governance. This experience gave me new insights into balancing innovation and ethics."

	fmt.Printf("\nRunning model inference:\n")
	fmt.Printf("Input: %s\n", inputText)

	startTime := time.Now()
	embeddings, err := embeddingModel.Embed(inputText)
	if err != nil {
		panic(err)
	}
	totalTime := time.Since(startTime)

	fmt.Printf("First inference time: %v\n", totalTime)
	fmt.Printf("First 10 values: %v\n", embeddings[:10])

	fmt.Printf("\nRunning second inference to show speed improvement:\n")
	startTime = time.Now()
	embeddings2, err := embeddingModel.Embed(inputText)
	if err != nil {
		panic(err)
	}
	totalTime2 := time.Since(startTime)
	fmt.Printf("Second inference time: %v\n", totalTime2)
	fmt.Printf("First 10 values: %v\n", embeddings2[:10])
}

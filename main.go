package main

import (
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	"math"
	"strings"
)

func meanPooling(modelOutput []float32, attentionMask []int64, batchSize, seqLen, embedDim int) []float32 {
	result := make([]float32, batchSize*embedDim)

	for b := 0; b < batchSize; b++ {
		var sumMask float32
		for i := 0; i < embedDim; i++ {
			var sumEmbedding float32
			for s := 0; s < seqLen; s++ {
				maskVal := float32(attentionMask[b*seqLen+s])
				embeddingVal := modelOutput[b*seqLen*embedDim+s*embedDim+i]
				sumEmbedding += embeddingVal * maskVal
				if i == 0 {
					sumMask += maskVal
				}
			}
			if sumMask < 1e-9 {
				sumMask = 1e-9
			}
			result[b*embedDim+i] = sumEmbedding / sumMask
		}
	}
	return result
}

func l2Normalize(embeddings []float32, batchSize, embedDim int) []float32 {
	result := make([]float32, len(embeddings))

	for b := 0; b < batchSize; b++ {
		var norm float32
		for i := 0; i < embedDim; i++ {
			val := embeddings[b*embedDim+i]
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))

		for i := 0; i < embedDim; i++ {
			result[b*embedDim+i] = embeddings[b*embedDim+i] / norm
		}
	}
	return result
}

func main() {
	lib := "/usr/local/lib/onnxruntime/lib/libonnxruntime.so"
	ort.SetSharedLibraryPath(lib)

	err := ort.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	defer ort.DestroyEnvironment()

	// Initialize real SentencePiece tokenizer
	tokenizer := NewSentencePieceTokenizer()
	err = tokenizer.LoadFromHuggingFace("jinaai/jina-embeddings-v3")
	if err != nil {
		panic(fmt.Errorf("failed to load tokenizer: %v", err))
	}

	// Test with different texts to show dynamic tokenization
	testTexts := []string{
		"This is an apple",
		"Hello world!",
		"机器学习很有趣",
		"The quick brown fox jumps over the lazy dog",
		"🚀 Tokenization is working!",
	}

	for _, text := range testTexts {
		fmt.Printf("\n" + strings.Repeat("=", 60) + "\n")
		fmt.Printf("Testing text: %s\n", text)
		ids, _ := tokenizer.Encode(text)
		fmt.Printf("Final result - IDs: %v\n", ids)
		fmt.Printf("Decoded: %s\n", tokenizer.DecodeIds(ids))
	}

	// Use the first text for the model
	inputText := testTexts[0]
	inputIds, attentionMask := tokenizer.Encode(inputText)

	// Get task ID dynamically
	taskType := "text-matching"
	taskIdValue, err := tokenizer.GetTaskID(taskType)
	if err != nil {
		panic(fmt.Errorf("failed to get task ID: %v", err))
	}
	taskId := []int64{taskIdValue}

	// Test decoding
	fmt.Printf("Decoded text: %s\n", tokenizer.DecodeIds(inputIds))

	// Dynamic dimensions based on tokenization
	batchSize := 1
	seqLen := len(inputIds)
	embedDim := 1024

	fmt.Printf("Input text: %s\n", inputText)
	fmt.Printf("Input IDs: %v\n", inputIds)
	fmt.Printf("Attention mask: %v\n", attentionMask)
	fmt.Printf("Task ID: %v\n", taskId)

	// Create input tensors
	inputIdsShape := ort.NewShape(int64(batchSize), int64(seqLen))
	inputIdsTensor, err := ort.NewTensor(inputIdsShape, inputIds)
	if err != nil {
		panic(err)
	}
	defer inputIdsTensor.Destroy()

	attentionMaskShape := ort.NewShape(int64(batchSize), int64(seqLen))
	attentionMaskTensor, err := ort.NewTensor(attentionMaskShape, attentionMask)
	if err != nil {
		panic(err)
	}
	defer attentionMaskTensor.Destroy()

	taskIdShape := ort.NewShape(1)
	taskIdTensor, err := ort.NewTensor(taskIdShape, taskId)
	if err != nil {
		panic(err)
	}
	defer taskIdTensor.Destroy()

	// Create output tensor
	outputShape := ort.NewShape(int64(batchSize), int64(seqLen), int64(embedDim))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		panic(err)
	}
	defer outputTensor.Destroy()

	model := "py/model/model.onnx"
	session, err := ort.NewAdvancedSession(model,
		[]string{"input_ids", "attention_mask", "task_id"},
		[]string{"text_embeds"},
		[]ort.Value{inputIdsTensor, attentionMaskTensor, taskIdTensor},
		[]ort.Value{outputTensor}, nil)
	if err != nil {
		panic(err)
	}
	defer session.Destroy()

	err = session.Run()
	if err != nil {
		panic(err)
	}

	// Get model output
	rawOutput := outputTensor.GetData()

	// Apply mean pooling
	pooledEmbeddings := meanPooling(rawOutput, attentionMask, batchSize, seqLen, embedDim)

	// Apply L2 normalization
	finalEmbeddings := l2Normalize(pooledEmbeddings, batchSize, embedDim)

	fmt.Printf("Final embeddings shape: [%d, %d]\n", batchSize, embedDim)
	fmt.Printf("First 10 values: %v\n", finalEmbeddings[:10])
}

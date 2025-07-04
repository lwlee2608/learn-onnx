package main

import (
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	"math"
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
	err = tokenizer.LoadFromHuggingFace("jinaai/jina-embeddings-v2-base-en")
	if err != nil {
		panic(fmt.Errorf("failed to load tokenizer: %v", err))
	}

	// Tokenize input text dynamically
	inputText := "This is an apple"
	inputIds, attentionMask := tokenizer.Encode(inputText)
	
	// Create token type IDs (all zeros for single sequence)
	tokenTypeIds := make([]int64, len(inputIds))
	for i := range tokenTypeIds {
		tokenTypeIds[i] = 0
	}

	// Get task ID dynamically (not used in this model)
	taskType := "text-matching"
	taskIdValue, err := tokenizer.GetTaskID(taskType)
	if err != nil {
		panic(fmt.Errorf("failed to get task ID: %v", err))
	}
	taskId := []int64{taskIdValue}

	// Dynamic dimensions based on tokenization
	batchSize := 1
	seqLen := len(inputIds)
	embedDim := 768

	fmt.Printf("\nRunning model inference:\n")
	fmt.Printf("Input: %s\n", inputText)
	fmt.Printf("Token IDs: %v\n", inputIds)
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

	tokenTypeIdsShape := ort.NewShape(int64(batchSize), int64(seqLen))
	tokenTypeIdsTensor, err := ort.NewTensor(tokenTypeIdsShape, tokenTypeIds)
	if err != nil {
		panic(err)
	}
	defer tokenTypeIdsTensor.Destroy()

	// taskIdShape := ort.NewShape(1)
	// taskIdTensor, err := ort.NewTensor(taskIdShape, taskId)
	// if err != nil {
	// 	panic(err)
	// }
	// defer taskIdTensor.Destroy()

	// Create output tensor
	outputShape := ort.NewShape(int64(batchSize), int64(seqLen), int64(embedDim))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		panic(err)
	}
	defer outputTensor.Destroy()

	model := "py/model/model.onnx"
	session, err := ort.NewAdvancedSession(model,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]ort.Value{inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor},
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

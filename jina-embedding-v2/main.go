package main

import (
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	"math"
	"runtime"
	"time"
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

type EmbeddingModel struct {
	session   *ort.DynamicAdvancedSession
	tokenizer *SentencePieceTokenizer
}

func NewEmbeddingModel(modelPath string) (*EmbeddingModel, error) {
	// Set library path based on OS
	switch runtime.GOOS {
	case "linux":
		ort.SetSharedLibraryPath("/usr/local/lib/onnxruntime/lib/libonnxruntime.so")
	case "darwin":
		ort.SetSharedLibraryPath("/usr/local/lib/onnxruntime/libonnxruntime.dylib")
	default:
		return nil, fmt.Errorf("unsupported operating system: %s", runtime.GOOS)
	}

	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, err
	}

	tokenizer := NewSentencePieceTokenizer()
	err = tokenizer.LoadFromHuggingFace("jinaai/jina-embeddings-v2-base-en")
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %v", err)
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"}, nil)
	if err != nil {
		return nil, err
	}

	return &EmbeddingModel{
		session:   session,
		tokenizer: tokenizer,
	}, nil
}

func (m *EmbeddingModel) Close() {
	if m.session != nil {
		m.session.Destroy()
	}
	ort.DestroyEnvironment()
}

func (m *EmbeddingModel) Embed(inputText string) ([]float32, error) {
	tokenizerStart := time.Now()
	inputIds, attentionMask := m.tokenizer.Encode(inputText)
	tokenizerTime := time.Since(tokenizerStart)

	tokenTypeIds := make([]int64, len(inputIds))
	for i := range tokenTypeIds {
		tokenTypeIds[i] = 0
	}

	batchSize := 1
	seqLen := len(inputIds)
	embedDim := 768

	tensorStart := time.Now()
	inputIdsShape := ort.NewShape(int64(batchSize), int64(seqLen))
	inputIdsTensor, err := ort.NewTensor(inputIdsShape, inputIds)
	if err != nil {
		return nil, err
	}
	defer inputIdsTensor.Destroy()

	attentionMaskShape := ort.NewShape(int64(batchSize), int64(seqLen))
	attentionMaskTensor, err := ort.NewTensor(attentionMaskShape, attentionMask)
	if err != nil {
		return nil, err
	}
	defer attentionMaskTensor.Destroy()

	tokenTypeIdsShape := ort.NewShape(int64(batchSize), int64(seqLen))
	tokenTypeIdsTensor, err := ort.NewTensor(tokenTypeIdsShape, tokenTypeIds)
	if err != nil {
		return nil, err
	}
	defer tokenTypeIdsTensor.Destroy()

	outputShape := ort.NewShape(int64(batchSize), int64(seqLen), int64(embedDim))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()
	tensorTime := time.Since(tensorStart)

	inferenceStart := time.Now()
	err = m.session.Run([]ort.Value{inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor}, []ort.Value{outputTensor})
	if err != nil {
		return nil, err
	}
	inferenceTime := time.Since(inferenceStart)

	fmt.Printf("Tokenizer time: %v\n", tokenizerTime)
	fmt.Printf("Tensor creation time: %v\n", tensorTime)
	fmt.Printf("Inference time: %v\n", inferenceTime)

	rawOutput := outputTensor.GetData()
	pooledEmbeddings := meanPooling(rawOutput, attentionMask, batchSize, seqLen, embedDim)
	finalEmbeddings := l2Normalize(pooledEmbeddings, batchSize, embedDim)

	return finalEmbeddings, nil
}

func main() {
	model := "model/model.onnx"

	fmt.Printf("Initializing embedding model...\n")
	initStart := time.Now()
	embeddingModel, err := NewEmbeddingModel(model)
	if err != nil {
		panic(err)
	}
	defer embeddingModel.Close()
	initTime := time.Since(initStart)
	fmt.Printf("Model initialization time: %v\n", initTime)

	inputText := "This is an apple"
	fmt.Printf("\nRunning model inference:\n")
	fmt.Printf("Input: %s\n", inputText)

	startTime := time.Now()
	embeddings, err := embeddingModel.Embed(inputText)
	if err != nil {
		panic(err)
	}
	totalTime := time.Since(startTime)

	fmt.Printf("Total inference time: %v\n", totalTime)
	fmt.Printf("Final embeddings shape: [%d, %d]\n", 1, 768)
	fmt.Printf("First 10 values: %v\n", embeddings[:10])

	fmt.Printf("\nRunning second inference to show speed improvement:\n")
	startTime = time.Now()
	embeddings2, err := embeddingModel.Embed("This is a second test")
	if err != nil {
		panic(err)
	}
	totalTime2 := time.Since(startTime)
	fmt.Printf("Second inference time: %v\n", totalTime2)
	fmt.Printf("First 10 values: %v\n", embeddings2[:10])
}

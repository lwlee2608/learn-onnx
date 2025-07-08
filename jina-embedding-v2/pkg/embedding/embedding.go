package embedding

import (
	"fmt"
	"math"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
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

type Tokenizer interface {
	Encode(text string) ([]int64, []int64)
}

type Model struct {
	session   *ort.DynamicAdvancedSession
	tokenizer Tokenizer
}

func NewModel(modelPath string, tokenizer Tokenizer) (*Model, error) {
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

	session, err := ort.NewDynamicAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"}, nil)
	if err != nil {
		return nil, err
	}

	return &Model{
		session:   session,
		tokenizer: tokenizer,
	}, nil
}

func (m *Model) Close() {
	if m.session != nil {
		m.session.Destroy()
	}
	ort.DestroyEnvironment()
}

func (m *Model) Embed(inputText string) ([]float32, error) {
	inputIds, attentionMask := m.tokenizer.Encode(inputText)

	tokenTypeIds := make([]int64, len(inputIds))
	for i := range tokenTypeIds {
		tokenTypeIds[i] = 0
	}

	batchSize := 1
	seqLen := len(inputIds)
	embedDim := 768

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

	err = m.session.Run([]ort.Value{inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor}, []ort.Value{outputTensor})
	if err != nil {
		return nil, err
	}

	rawOutput := outputTensor.GetData()
	pooledEmbeddings := meanPooling(rawOutput, attentionMask, batchSize, seqLen, embedDim)
	finalEmbeddings := l2Normalize(pooledEmbeddings, batchSize, embedDim)

	return finalEmbeddings, nil
}
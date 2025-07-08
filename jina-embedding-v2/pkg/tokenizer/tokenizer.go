package tokenizer

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type ModelConfig struct {
	LoraAdaptations []string `json:"lora_adaptations"`
}

type SentencePieceTokenizer struct {
	vocab         map[string]int
	vocabReverse  map[int]string
	specialTokens map[string]int
	config        *ModelConfig
	bosToken      string
	eosToken      string
	unkToken      string
}

type TokenizerJSON struct {
	Version string `json:"version"`
	Model   struct {
		Type       string      `json:"type"`
		Vocab      interface{} `json:"vocab"`
		UnkId      int         `json:"unk_id"`
		Dropout    *float64    `json:"dropout"`
		Continuing interface{} `json:"continuing_subword_prefix"`
		EndOfWord  bool        `json:"end_of_word_suffix"`
		FuseUnk    bool        `json:"fuse_unk"`
	} `json:"model"`
	Normalizer struct {
		Type string `json:"type"`
	} `json:"normalizer"`
	PreTokenizer struct {
		Type       string `json:"type"`
		AddPrefix  bool   `json:"add_prefix_space"`
		TrimOffset bool   `json:"trim_offsets"`
	} `json:"pre_tokenizer"`
	PostProcessor struct {
		Type string   `json:"type"`
		Sep  []string `json:"sep"`
		Cls  []string `json:"cls"`
	} `json:"post_processor"`
	Decoder struct {
		Type string `json:"type"`
	} `json:"decoder"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

func NewSentencePieceTokenizer() *SentencePieceTokenizer {
	return &SentencePieceTokenizer{
		vocab:         make(map[string]int),
		vocabReverse:  make(map[int]string),
		specialTokens: make(map[string]int),
		bosToken:      "<s>",
		eosToken:      "</s>",
		unkToken:      "<unk>",
	}
}

func (t *SentencePieceTokenizer) LoadFromLocal(tokenizerPath, configPath string) error {
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		return fmt.Errorf("tokenizer.json not found at %s", tokenizerPath)
	}

	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return fmt.Errorf("config.json not found at %s", configPath)
	}

	tokenizerData, err := os.ReadFile(tokenizerPath)
	if err != nil {
		return fmt.Errorf("failed to read tokenizer.json: %v", err)
	}

	var tokenizerJSON TokenizerJSON
	err = json.Unmarshal(tokenizerData, &tokenizerJSON)
	if err != nil {
		return fmt.Errorf("failed to parse tokenizer.json: %v", err)
	}

	configData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config.json: %v", err)
	}

	var modelConfig ModelConfig
	err = json.Unmarshal(configData, &modelConfig)
	if err != nil {
		return fmt.Errorf("failed to parse config.json: %v", err)
	}

	t.config = &modelConfig

	switch vocab := tokenizerJSON.Model.Vocab.(type) {
	case map[string]interface{}:
		for token, id := range vocab {
			if idInt, ok := id.(float64); ok {
				t.vocab[token] = int(idInt)
				t.vocabReverse[int(idInt)] = token
			}
		}
	case []interface{}:
		for i, vocabItem := range vocab {
			if vocabArray, ok := vocabItem.([]interface{}); ok && len(vocabArray) >= 2 {
				if token, ok := vocabArray[0].(string); ok {
					t.vocab[token] = i
					t.vocabReverse[i] = token
				}
			}
		}
	}

	for _, token := range tokenizerJSON.AddedTokens {
		t.specialTokens[token.Content] = token.ID
		switch token.Content {
		case "<s>":
			t.bosToken = token.Content
		case "</s>":
			t.eosToken = token.Content
		case "<unk>":
			t.unkToken = token.Content
		}
	}

	fmt.Printf("Loaded tokenizer with vocab size: %d\n", len(t.vocab))
	fmt.Printf("Special tokens: %v\n", t.specialTokens)

	return nil
}

func (t *SentencePieceTokenizer) LoadFromHuggingFace(modelName string) error {
	baseURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main", modelName)

	cacheDir := filepath.Join(os.TempDir(), "real_tokenizer_cache", modelName)
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return fmt.Errorf("failed to create cache directory: %v", err)
	}

	tokenizerPath := filepath.Join(cacheDir, "tokenizer.json")
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		fmt.Printf("Downloading tokenizer.json...\n")
		err := t.downloadFile(baseURL+"/tokenizer.json", tokenizerPath)
		if err != nil {
			return fmt.Errorf("failed to download tokenizer.json: %v", err)
		}
	}

	configPath := filepath.Join(cacheDir, "config.json")
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		fmt.Printf("Downloading config.json...\n")
		err := t.downloadFile(baseURL+"/config.json", configPath)
		if err != nil {
			return fmt.Errorf("failed to download config.json: %v", err)
		}
	}

	tokenizerData, err := os.ReadFile(tokenizerPath)
	if err != nil {
		return fmt.Errorf("failed to read tokenizer.json: %v", err)
	}

	var tokenizerJSON TokenizerJSON
	err = json.Unmarshal(tokenizerData, &tokenizerJSON)
	if err != nil {
		return fmt.Errorf("failed to parse tokenizer.json: %v", err)
	}

	configData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config.json: %v", err)
	}

	var modelConfig ModelConfig
	err = json.Unmarshal(configData, &modelConfig)
	if err != nil {
		return fmt.Errorf("failed to parse config.json: %v", err)
	}

	t.config = &modelConfig

	switch vocab := tokenizerJSON.Model.Vocab.(type) {
	case map[string]interface{}:
		for token, id := range vocab {
			if idInt, ok := id.(float64); ok {
				t.vocab[token] = int(idInt)
				t.vocabReverse[int(idInt)] = token
			}
		}
	case []interface{}:
		for i, vocabItem := range vocab {
			if vocabArray, ok := vocabItem.([]interface{}); ok && len(vocabArray) >= 2 {
				if token, ok := vocabArray[0].(string); ok {
					t.vocab[token] = i
					t.vocabReverse[i] = token
				}
			}
		}
	}

	for _, token := range tokenizerJSON.AddedTokens {
		t.specialTokens[token.Content] = token.ID
		switch token.Content {
		case "<s>":
			t.bosToken = token.Content
		case "</s>":
			t.eosToken = token.Content
		case "<unk>":
			t.unkToken = token.Content
		}
	}

	fmt.Printf("Loaded tokenizer with vocab size: %d\n", len(t.vocab))
	fmt.Printf("Special tokens: %v\n", t.specialTokens)

	return nil
}

func (t *SentencePieceTokenizer) downloadFile(url, filepath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer func() {
		if err := resp.Body.Close(); err != nil {
			fmt.Printf("Warning: failed to close response body: %v\n", err)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file: status %d", resp.StatusCode)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer func() {
		if err := out.Close(); err != nil {
			fmt.Printf("Warning: failed to close file: %v\n", err)
		}
	}()

	_, err = io.Copy(out, resp.Body)
	return err
}

func (t *SentencePieceTokenizer) tokenToIds(tokens []string) []int64 {
	var ids []int64
	for _, token := range tokens {
		if id, exists := t.vocab[token]; exists {
			ids = append(ids, int64(id))
		} else {
			if id, exists := t.specialTokens[token]; exists {
				ids = append(ids, int64(id))
			} else {
				ids = append(ids, int64(t.specialTokens[t.unkToken]))
			}
		}
	}
	return ids
}

func (t *SentencePieceTokenizer) Encode(text string) ([]int64, []int64) {
	text = strings.ToLower(text)
	words := strings.Fields(text)

	var tokens []string
	tokens = append(tokens, "[CLS]")
	tokens = append(tokens, words...)
	tokens = append(tokens, "[SEP]")

	inputIds := t.tokenToIds(tokens)

	attentionMask := make([]int64, len(inputIds))
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	return inputIds, attentionMask
}

func (t *SentencePieceTokenizer) GetTaskID(taskType string) (int64, error) {
	if t.config == nil {
		return 0, fmt.Errorf("config not loaded")
	}

	for i, task := range t.config.LoraAdaptations {
		if task == taskType {
			return int64(i), nil
		}
	}

	return 0, nil
}

func (t *SentencePieceTokenizer) DecodeIds(ids []int64) string {
	var tokens []string
	for _, id := range ids {
		if token, exists := t.vocabReverse[int(id)]; exists {
			tokens = append(tokens, token)
		} else {
			tokens = append(tokens, t.unkToken)
		}
	}

	text := strings.Join(tokens, "")
	text = strings.ReplaceAll(text, "▁", " ")
	text = strings.ReplaceAll(text, t.bosToken, "")
	text = strings.ReplaceAll(text, t.eosToken, "")

	return strings.TrimSpace(text)
}
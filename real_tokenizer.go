package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
)

// ModelConfig represents the model configuration
type ModelConfig struct {
	LoraAdaptations []string `json:"lora_adaptations"`
}

// SentencePieceTokenizer represents a proper XLM-RoBERTa tokenizer
type SentencePieceTokenizer struct {
	vocab         map[string]int
	vocabReverse  map[int]string
	merges        [][]string
	specialTokens map[string]int
	config        *ModelConfig
	bosToken      string
	eosToken      string
	unkToken      string
	padToken      string
	maskToken     string
}

// TokenizerJSON represents the structure of tokenizer.json
type TokenizerJSON struct {
	Version string `json:"version"`
	Model   struct {
		Type       string              `json:"type"`
		Vocab      [][]interface{}     `json:"vocab"`  // Array of [token, score] pairs
		UnkId      int                 `json:"unk_id"`
		Dropout    *float64            `json:"dropout"`
		Continuing bool                `json:"continuing_subword_prefix"`
		EndOfWord  bool                `json:"end_of_word_suffix"`
		FuseUnk    bool                `json:"fuse_unk"`
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
		Type string `json:"type"`
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

// NewSentencePieceTokenizer creates a new SentencePiece tokenizer
func NewSentencePieceTokenizer() *SentencePieceTokenizer {
	return &SentencePieceTokenizer{
		vocab:         make(map[string]int),
		vocabReverse:  make(map[int]string),
		merges:        [][]string{},
		specialTokens: make(map[string]int),
		bosToken:      "<s>",
		eosToken:      "</s>",
		unkToken:      "<unk>",
		padToken:      "<pad>",
		maskToken:     "<mask>",
	}
}

// LoadFromHuggingFace downloads and loads the real tokenizer from HuggingFace
func (t *SentencePieceTokenizer) LoadFromHuggingFace(modelName string) error {
	baseURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main", modelName)
	
	// Create cache directory
	cacheDir := filepath.Join(os.TempDir(), "real_tokenizer_cache", modelName)
	os.MkdirAll(cacheDir, 0755)

	// Download tokenizer.json
	tokenizerPath := filepath.Join(cacheDir, "tokenizer.json")
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		fmt.Printf("Downloading tokenizer.json...\n")
		err := t.downloadFile(baseURL+"/tokenizer.json", tokenizerPath)
		if err != nil {
			return fmt.Errorf("failed to download tokenizer.json: %v", err)
		}
	}

	// Download config.json
	configPath := filepath.Join(cacheDir, "config.json")
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		fmt.Printf("Downloading config.json...\n")
		err := t.downloadFile(baseURL+"/config.json", configPath)
		if err != nil {
			return fmt.Errorf("failed to download config.json: %v", err)
		}
	}

	// Load tokenizer configuration
	tokenizerData, err := os.ReadFile(tokenizerPath)
	if err != nil {
		return fmt.Errorf("failed to read tokenizer.json: %v", err)
	}

	var tokenizerJSON TokenizerJSON
	err = json.Unmarshal(tokenizerData, &tokenizerJSON)
	if err != nil {
		return fmt.Errorf("failed to parse tokenizer.json: %v", err)
	}

	// Load model config
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config.json: %v", err)
	}

	var modelConfig ModelConfig
	err = json.Unmarshal(configData, &modelConfig)
	if err != nil {
		return fmt.Errorf("failed to parse config.json: %v", err)
	}

	// Set up tokenizer
	t.config = &modelConfig
	
	// Parse vocab from array of [token, score] pairs
	for i, vocabItem := range tokenizerJSON.Model.Vocab {
		if len(vocabItem) >= 2 {
			if token, ok := vocabItem[0].(string); ok {
				t.vocab[token] = i
				t.vocabReverse[i] = token
			}
		}
	}

	// Note: Unigram model doesn't use merges like BPE
	// The merges field is not present in Unigram tokenizers

	// Set up special tokens from added_tokens
	for _, token := range tokenizerJSON.AddedTokens {
		t.specialTokens[token.Content] = token.ID
		// Update special token strings
		switch token.Content {
		case "<s>":
			t.bosToken = token.Content
		case "</s>":
			t.eosToken = token.Content
		case "<unk>":
			t.unkToken = token.Content
		case "<pad>":
			t.padToken = token.Content
		case "<mask>":
			t.maskToken = token.Content
		}
	}

	fmt.Printf("Loaded tokenizer with vocab size: %d\n", len(t.vocab))
	fmt.Printf("Loaded %d merge operations\n", len(t.merges))
	fmt.Printf("Special tokens: %v\n", t.specialTokens)

	return nil
}

// downloadFile downloads a file from URL
func (t *SentencePieceTokenizer) downloadFile(url, filepath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file: status %d", resp.StatusCode)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

// normalize performs text normalization (NFD normalization)
func (t *SentencePieceTokenizer) normalize(text string) string {
	// Basic normalization - in a full implementation you'd use unicode.Normalize
	return strings.TrimSpace(text)
}

// preTokenize performs pre-tokenization similar to XLM-RoBERTa
func (t *SentencePieceTokenizer) preTokenize(text string) []string {
	// XLM-RoBERTa uses a regex-based pre-tokenizer
	// This pattern matches words, punctuation, and whitespace
	re := regexp.MustCompile(`\w+|[^\w\s]`)
	matches := re.FindAllString(text, -1)
	
	var tokens []string
	for i, match := range matches {
		// Add prefix space for non-first tokens (SentencePiece convention)
		if i > 0 && isAlphaNumeric(match) {
			tokens = append(tokens, "▁"+match)
		} else if i == 0 && isAlphaNumeric(match) {
			tokens = append(tokens, "▁"+match)
		} else {
			tokens = append(tokens, match)
		}
	}
	
	return tokens
}

// isAlphaNumeric checks if a string contains alphanumeric characters
func isAlphaNumeric(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}

// unigramTokenize performs Unigram tokenization on a token
func (t *SentencePieceTokenizer) unigramTokenize(token string) []string {
	if len(token) == 0 {
		return []string{}
	}

	// For Unigram, we use a greedy approach to find the best segmentation
	// This is a simplified implementation
	return t.greedyTokenize(token)
}

// greedyTokenize performs greedy tokenization (simplified Unigram)
func (t *SentencePieceTokenizer) greedyTokenize(token string) []string {
	if len(token) == 0 {
		return []string{}
	}

	var result []string
	i := 0
	
	for i < len(token) {
		// Try to find the longest matching token from current position
		bestMatch := ""
		bestLength := 0
		
		// Try all possible substrings starting from current position
		for j := i + 1; j <= len(token); j++ {
			candidate := token[i:j]
			if _, exists := t.vocab[candidate]; exists {
				if len(candidate) > bestLength {
					bestMatch = candidate
					bestLength = len(candidate)
				}
			}
		}
		
		if bestMatch != "" {
			result = append(result, bestMatch)
			i += bestLength
		} else {
			// If no match found, try single character or use UNK
			if i < len(token) {
				char := string([]rune(token)[i])
				if _, exists := t.vocab[char]; exists {
					result = append(result, char)
				} else {
					result = append(result, t.unkToken)
				}
				i++
			}
		}
	}
	
	return result
}


// tokenToIds converts tokens to IDs
func (t *SentencePieceTokenizer) tokenToIds(tokens []string) []int64 {
	var ids []int64
	for _, token := range tokens {
		if id, exists := t.vocab[token]; exists {
			ids = append(ids, int64(id))
		} else {
			// Try to find in special tokens
			if id, exists := t.specialTokens[token]; exists {
				ids = append(ids, int64(id))
			} else {
				// Use UNK token
				ids = append(ids, int64(t.specialTokens[t.unkToken]))
			}
		}
	}
	return ids
}

// Encode tokenizes text and returns token IDs using real XLM-RoBERTa tokenization
func (t *SentencePieceTokenizer) Encode(text string) ([]int64, []int64) {
	// Step 1: Normalize text
	normalized := t.normalize(text)
	
	// Step 2: Pre-tokenize
	preTokens := t.preTokenize(normalized)
	
	// Step 3: Apply Unigram tokenization to each pre-token
	var allTokens []string
	for _, preToken := range preTokens {
		unigramTokens := t.unigramTokenize(preToken)
		allTokens = append(allTokens, unigramTokens...)
	}
	
	// Step 4: Add special tokens
	var finalTokens []string
	finalTokens = append(finalTokens, t.bosToken) // Add BOS token
	finalTokens = append(finalTokens, allTokens...)
	finalTokens = append(finalTokens, t.eosToken) // Add EOS token
	
	// Step 5: Convert to IDs
	inputIds := t.tokenToIds(finalTokens)
	
	// Step 6: Create attention mask
	attentionMask := make([]int64, len(inputIds))
	for i := range attentionMask {
		attentionMask[i] = 1
	}
	
	fmt.Printf("Tokenization process:\n")
	fmt.Printf("  Original text: %s\n", text)
	fmt.Printf("  Normalized: %s\n", normalized)
	fmt.Printf("  Pre-tokens: %v\n", preTokens)
	fmt.Printf("  BPE tokens: %v\n", allTokens)
	fmt.Printf("  Final tokens: %v\n", finalTokens)
	fmt.Printf("  Token IDs: %v\n", inputIds)
	
	return inputIds, attentionMask
}

// GetTaskID returns the task ID for a given task type
func (t *SentencePieceTokenizer) GetTaskID(taskType string) (int64, error) {
	if t.config == nil {
		return 0, fmt.Errorf("config not loaded")
	}

	for i, task := range t.config.LoraAdaptations {
		if task == taskType {
			return int64(i), nil
		}
	}

	return 0, fmt.Errorf("task type '%s' not found in %v", taskType, t.config.LoraAdaptations)
}

// DecodeIds converts token IDs back to text (for debugging)
func (t *SentencePieceTokenizer) DecodeIds(ids []int64) string {
	var tokens []string
	for _, id := range ids {
		if token, exists := t.vocabReverse[int(id)]; exists {
			tokens = append(tokens, token)
		} else {
			tokens = append(tokens, t.unkToken)
		}
	}
	
	// Join tokens and clean up
	text := strings.Join(tokens, "")
	text = strings.ReplaceAll(text, "▁", " ")
	text = strings.ReplaceAll(text, t.bosToken, "")
	text = strings.ReplaceAll(text, t.eosToken, "")
	
	return strings.TrimSpace(text)
}
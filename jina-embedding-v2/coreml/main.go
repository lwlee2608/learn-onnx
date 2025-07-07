package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

func main() {
	binaryPath := "./coreml-cli-v2"
	modelPath := "./jina-v2"
	// input := "This is an apple"
	input := "On the morning of April 16, 2024, I attended the annual AI Innovation Conference in downtown San Francisco. The keynote speaker, Dr. Evelyn Chen, discussed the ethical implications of autonomous decision-making systems in healthcare. I remember the room was filled with experts from various fields, including data science, medicine, and law. After her talk, I had a conversation with a software engineer named Miguel who was developing a diagnostic tool powered by GPT-4. He shared insights about real-world challenges in gathering unbiased medical data. Later, I participated in a roundtable about data privacy and shared my perspective on how granular access controls could help protect sensitive patient information. The day ended with a networking session where I met professionals interested in AI governance. This experience gave me new insights into balancing innovation and ethics."
	service := NewService(binaryPath, modelPath, true)
	defer service.Close()

	start := time.Now()
	_, err := service.Infer(input)
	elapsed := time.Since(start)

	// fmt.Printf("Result: %s", result[10:])
	if err != nil {
		fmt.Printf("Error: %v", err)
	}
	fmt.Printf("\nInteractive inference time: %v", elapsed)
}

// Service.go (TOBE the service to be interacted with)
type Service struct {
	binaryPath  string
	modelPath   string
	interactive bool
	cmd         *exec.Cmd
	stdin       io.WriteCloser
	stdout      io.ReadCloser
	scanner     *bufio.Scanner
	mu          sync.Mutex
}

func NewService(binaryPath, modelPath string, interactive bool) *Service {
	s := &Service{
		binaryPath:  binaryPath,
		modelPath:   modelPath,
		interactive: interactive,
	}

	if interactive {
		if err := s.startInteractiveProcess(); err != nil {
			s.interactive = false
		}
	}

	return s
}

func (s *Service) Infer(inputValue string) (string, error) {
	if s.interactive {
		return s.inferInteractive(inputValue)
	}
	return s.inferNonInteractive(inputValue)
}

func (s *Service) inferInteractive(inputValue string) (string, error) {
	fmt.Printf("inferencing : %s", inputValue)
	s.mu.Lock()
	defer s.mu.Unlock()

	for retries := 0; retries < 2; retries++ {
		if s.cmd == nil || s.stdin == nil || s.scanner == nil {
			if err := s.restartInteractiveProcess(); err != nil {
				if retries == 1 {
					return "", fmt.Errorf("failed to restart interactive process: %w", err)
				}
				continue
			}
		}

		if s.cmd.ProcessState != nil && s.cmd.ProcessState.Exited() {
			if err := s.restartInteractiveProcess(); err != nil {
				if retries == 1 {
					return "", fmt.Errorf("failed to restart interactive process after exit: %w", err)
				}
				continue
			}
		}

		input := map[string]interface{}{
			"inputs": []string{inputValue},
		}
		inputJSON, err := json.Marshal(input)
		if err != nil {
			return "", fmt.Errorf("failed to marshal input JSON: %w", err)
		}

		if _, err := s.stdin.Write(append(inputJSON, '\n')); err != nil {
			if retries < 1 {
				s.restartInteractiveProcess()
				continue
			}
			return "", fmt.Errorf("failed to write to stdin: %w", err)
		}

		if !s.scanner.Scan() {
			if err := s.scanner.Err(); err != nil {
				if retries < 1 {
					s.restartInteractiveProcess()
					continue
				}
				return "", fmt.Errorf("failed to read from stdout: %w", err)
			}
			return "", fmt.Errorf("no response from interactive process")
		}

		response := strings.TrimSpace(s.scanner.Text())
		return response, nil
	}

	return "", fmt.Errorf("failed to get response after retries")
}

func (s *Service) inferNonInteractive(inputValue string) (string, error) {
	if _, err := os.Stat(s.binaryPath); os.IsNotExist(err) {
		return "", fmt.Errorf("coreml-cli binary not found at %s", s.binaryPath)
	}

	if _, err := os.Stat(s.modelPath); os.IsNotExist(err) {
		return "", fmt.Errorf("model not found at %s", s.modelPath)
	}

	cmd := exec.Command(s.binaryPath, "infer", s.modelPath, inputValue)

	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to execute coreml-cli: %w, output: %s", err, string(output))
	}

	return string(output), nil
}

func (s *Service) startInteractiveProcess() error {
	if _, err := os.Stat(s.binaryPath); os.IsNotExist(err) {
		return fmt.Errorf("coreml-cli binary not found at %s", s.binaryPath)
	}

	if _, err := os.Stat(s.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("model not found at %s", s.modelPath)
	}

	s.cmd = exec.Command(s.binaryPath, "interactive", s.modelPath)

	stdin, err := s.cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdin pipe: %w", err)
	}
	s.stdin = stdin

	stdout, err := s.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}
	s.stdout = stdout
	s.scanner = bufio.NewScanner(stdout)

	// Set a larger buffer size to handle large embedding responses
	buf := make([]byte, 10*1024*1024) // 10MB buffer
	s.scanner.Buffer(buf, 10*1024*1024)

	if err := s.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start interactive process: %w", err)
	}

	return nil
}

func (s *Service) stopInteractiveProcess() error {
	if s.cmd == nil {
		return nil
	}

	if s.stdin != nil {
		s.stdin.Close()
	}
	if s.stdout != nil {
		s.stdout.Close()
	}

	if s.cmd.Process != nil {
		if err := s.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill process: %w", err)
		}
	}

	s.cmd = nil
	s.stdin = nil
	s.stdout = nil
	s.scanner = nil

	return nil
}

func (s *Service) restartInteractiveProcess() error {
	fmt.Print("restarted")
	s.stopInteractiveProcess()
	return s.startInteractiveProcess()
}

func (s *Service) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.interactive {
		return s.stopInteractiveProcess()
	}
	return nil
}

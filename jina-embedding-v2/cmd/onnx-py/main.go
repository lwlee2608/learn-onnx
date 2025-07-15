package main

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

const serverPort = "8888"

type InferenceRequest struct {
	Command string `json:"command"`
	Text    string `json:"text"`
}

type InferenceResponse struct {
	Embedding     []float64 `json:"embedding"`
	Shape         []int     `json:"shape"`
	InferenceTime float64   `json:"inference_time"`
	Error         string    `json:"error"`
}

func isServerRunning() bool {
	conn, err := net.Dial("tcp", "localhost:"+serverPort)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

func sendInferenceRequest(text string) (*InferenceResponse, error) {
	conn, err := net.Dial("tcp", "localhost:"+serverPort)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	request := InferenceRequest{
		Command: "infer",
		Text:    text,
	}

	requestData, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	_, err = conn.Write(requestData)
	if err != nil {
		return nil, err
	}

	buffer := make([]byte, 65536) // 64KB buffer for large embeddings
	n, err := conn.Read(buffer)
	if err != nil {
		return nil, err
	}

	var response InferenceResponse
	err = json.Unmarshal(buffer[:n], &response)
	if err != nil {
		return nil, err
	}

	return &response, nil
}

func sendShutdownRequest() error {
	conn, err := net.Dial("tcp", "localhost:"+serverPort)
	if err != nil {
		return err
	}
	defer conn.Close()

	request := InferenceRequest{
		Command: "shutdown",
		Text:    "",
	}

	requestData, err := json.Marshal(request)
	if err != nil {
		return err
	}

	_, err = conn.Write(requestData)
	if err != nil {
		return err
	}

	// Read response but don't wait long
	conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	buffer := make([]byte, 1024)
	_, err = conn.Read(buffer)
	// Ignore read errors as server might close connection
	return nil
}

func gracefulShutdown(serverCmd *exec.Cmd) {
	if serverCmd == nil {
		return
	}

	fmt.Println("Initiating graceful shutdown...")
	
	// Try to send shutdown command to server
	if err := sendShutdownRequest(); err != nil {
		fmt.Printf("Could not send shutdown request: %v\n", err)
	}
	
	// Wait a bit for graceful shutdown
	done := make(chan error, 1)
	go func() {
		done <- serverCmd.Wait()
	}()
	
	select {
	case <-done:
		fmt.Println("Server shut down gracefully")
	case <-time.After(5 * time.Second):
		fmt.Println("Timeout waiting for graceful shutdown, force killing...")
		serverCmd.Process.Kill()
		serverCmd.Wait()
	}
}

func startServer(pyDir string) *exec.Cmd {
	cmd := exec.Command("uv", "run", "main.py")
	cmd.Dir = pyDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	err := cmd.Start()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		return nil
	}

	return cmd
}

func main() {
	// Get the current working directory
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting current working directory: %v\n", err)
		os.Exit(1)
	}

	// Calculate the path to the py directory
	pyDir := filepath.Join(cwd, "py")
	pyDir, err = filepath.Abs(pyDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error resolving py directory path: %v\n", err)
		os.Exit(1)
	}

	// Check if py directory exists
	if _, err := os.Stat(pyDir); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: py directory not found at %s\n", pyDir)
		os.Exit(1)
	}

	// Check if main.py exists in py directory
	mainPyPath := filepath.Join(pyDir, "main.py")
	if _, err := os.Stat(mainPyPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: main.py not found at %s\n", mainPyPath)
		os.Exit(1)
	}

	var serverCmd *exec.Cmd
	serverStartTime := time.Now()

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Check if server is already running
	if !isServerRunning() {
		fmt.Println("Starting server and loading model...")
		serverCmd = startServer(pyDir)
		if serverCmd == nil {
			os.Exit(1)
		}

		// Set up signal handler for server cleanup
		go func() {
			<-sigChan
			gracefulShutdown(serverCmd)
			os.Exit(0)
		}()

		// Wait for server to start and load model
		fmt.Print("Waiting for server to be ready")
		for i := 0; i < 30; i++ { // Wait up to 30 seconds
			time.Sleep(1 * time.Second)
			fmt.Print(".")
			if isServerRunning() {
				break
			}
		}
		fmt.Println()

		if !isServerRunning() {
			fmt.Fprintf(os.Stderr, "Server failed to start within timeout\n")
			gracefulShutdown(serverCmd)
			os.Exit(1)
		}
	} else {
		fmt.Println("Server already running, using existing instance")
		// Set up signal handler for clean exit
		go func() {
			<-sigChan
			fmt.Println("Received shutdown signal, exiting...")
			os.Exit(0)
		}()
	}

	serverLoadDuration := time.Since(serverStartTime)
	fmt.Printf("Server setup time: %v\n", serverLoadDuration)

	// Run inference with hardcoded text
	inputText := "This is an apple"
	fmt.Printf("\nRunning inference with text: %s\n", inputText)

	start := time.Now()
	response, err := sendInferenceRequest(inputText)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error sending inference request: %v\n", err)
		if serverCmd != nil {
			serverCmd.Process.Kill()
		}
		os.Exit(1)
	}
	inferDuration := time.Since(start)

	if response.Error != "" {
		fmt.Fprintf(os.Stderr, "Inference error: %s\n", response.Error)
		if serverCmd != nil {
			serverCmd.Process.Kill()
		}
		os.Exit(1)
	}

	fmt.Printf("Input: %s\n", inputText)
	fmt.Printf("Python inference time: %.4f seconds\n", response.InferenceTime)
	fmt.Printf("Go inference time (including network): %v\n", inferDuration)
	fmt.Printf("Embedding shape: %v\n", response.Shape)
	fmt.Printf("First 10 values: %v\n", response.Embedding[:10])

	fmt.Printf("Total execution time: %v\n", serverLoadDuration+inferDuration)

	// Clean up server if we started it
	if serverCmd != nil {
		gracefulShutdown(serverCmd)
	}
}

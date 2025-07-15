package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

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

	// Create command to run uv run main.py in the py directory
	cmd := exec.Command("uv", "run", "main.py")
	cmd.Dir = pyDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin

	// Run the command and measure execution time
	start := time.Now()
	if err := cmd.Run(); err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			os.Exit(exitError.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "Error running Python script: %v\n", err)
		os.Exit(1)
	}
	duration := time.Since(start)
	
	fmt.Printf("Python script execution time: %v\n", duration)
}

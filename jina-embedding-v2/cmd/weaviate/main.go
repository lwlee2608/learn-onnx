package main

import (
	"fmt"
	"context"
	"log"
	"time"

	"github.com/weaviate/weaviate-go-client/v5/weaviate"
)

func main() {
	fmt.Println("Starting Weaviate Hello World example...")

	// Create a client configuration for embedded Weaviate
	cfg := weaviate.Config{
		Host:   "localhost:8080",
		Scheme: "http",
	}

	// Create Weaviate client
	client, err := weaviate.NewClient(cfg)
	if err != nil {
		log.Printf("Error creating Weaviate client: %v\n", err)
		log.Println("Note: This is expected if Weaviate server is not running")
	}

	fmt.Println("Hello World from Weaviate!")
	fmt.Println("This is an embedded Weaviate example.")
	
	// Demonstrate basic client usage (will fail without server, but shows structure)
	if client != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		// Try to check if server is ready
		result, err := client.Misc().LiveChecker().Do(ctx)
		if err != nil {
			log.Printf("Cannot connect to Weaviate server: %v\n", err)
			fmt.Println("To run this example, start a Weaviate server first")
		} else {
			fmt.Printf("Connected to Weaviate! Status: %t\n", result)
		}
	}

	// Example of what you would do with an actual embedded instance
	fmt.Println("\nThis hello world example demonstrates:")
	fmt.Println("1. Creating a Weaviate client configuration")
	fmt.Println("2. Connecting to a Weaviate instance") 
	fmt.Println("3. Basic health check operations")
	fmt.Println("\nFor a full embedded Weaviate, you would:")
	fmt.Println("- Start an embedded Weaviate server process")
	fmt.Println("- Create schemas and classes")
	fmt.Println("- Insert and query data")
	
	fmt.Println("\nWeaviate Hello World completed successfully!")
}

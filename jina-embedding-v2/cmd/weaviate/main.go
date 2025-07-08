package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/go-openapi/loads"
	"github.com/jessevdk/go-flags"
	"github.com/pkg/errors"
	"github.com/weaviate/weaviate-go-client/v5/weaviate"
	"github.com/weaviate/weaviate/adapters/handlers/rest"
	"github.com/weaviate/weaviate/adapters/handlers/rest/operations"
)

func main() {
	fmt.Println("Starting Weaviate Hello World with embedded server...")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start embedded Weaviate server
	server, err := BootstrapWeaviateServer(ctx, "8080", "./weaviate-data")
	if err != nil {
		fmt.Printf("Failed to start Weaviate server: %v\n", err)
		return
	}
	defer server.Shutdown()

	// Create client to connect to our embedded server
	cfg := weaviate.Config{
		Host:   "localhost:8080",
		Scheme: "http",
	}

	client, err := weaviate.NewClient(cfg)
	if err != nil {
		fmt.Printf("Error creating Weaviate client: %v\n", err)
		return
	}

	fmt.Println("Hello World from embedded Weaviate!")
	
	// Test connection to embedded server
	result, err := client.Misc().LiveChecker().Do(ctx)
	if err != nil {
		fmt.Printf("Cannot connect to Weaviate server: %v\n", err)
		return
	}
	
	fmt.Printf("Connected to embedded Weaviate! Status: %t\n", result)
	
	// Get cluster status
	cluster, err := client.Cluster().NodesStatusGetter().Do(ctx)
	if err != nil {
		fmt.Printf("Error getting cluster status: %v\n", err)
	} else {
		fmt.Printf("Cluster nodes: %d\n", len(cluster.Nodes))
	}

	fmt.Println("\nEmbedded Weaviate server is running successfully!")
	fmt.Println("Server will continue running until program exits...")
	
	// Keep the program running to demonstrate the server is working
	fmt.Println("Press Ctrl+C to stop the server")
	select {
	case <-ctx.Done():
		fmt.Println("Context canceled, shutting down...")
	}
}

func BootstrapWeaviateServer(ctx context.Context, port string, dataPath string) (*rest.Server, error) {
	// Set environment variables for Weaviate configuration
	_ = os.Setenv("CLUSTER_HOSTNAME", "node1")
	_ = os.Setenv("CLUSTER_GOSSIP_BIND_PORT", "7946")
	_ = os.Setenv("CLUSTER_DATA_BIND_PORT", "7947")
	_ = os.Unsetenv("CLUSTER_JOIN")
	_ = os.Setenv("DISABLE_TELEMETRY", "true")
	_ = os.Setenv("AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED", "true")
	_ = os.Setenv("AUTHORIZATION_ADMIN_LIST_ENABLED", "false")
	_ = os.Setenv("LOG_LEVEL", "info")

	startTime := time.Now()
	fmt.Printf("Starting Weaviate server bootstrap (port: %s, dataPath: %s)\n", port, dataPath)

	// Create data directory if it doesn't exist
	if _, err := os.Stat(dataPath); os.IsNotExist(err) {
		fmt.Printf("Creating Weaviate data directory: %s\n", dataPath)
		if err := os.MkdirAll(dataPath, 0o755); err != nil {
			return nil, errors.Wrap(err, "Failed to create Weaviate data directory")
		}
	}

	// Set persistence data path
	err := os.Setenv("PERSISTENCE_DATA_PATH", dataPath)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to set PERSISTENCE_DATA_PATH")
	}

	// Load swagger specification
	swaggerSpec, err := loads.Embedded(rest.SwaggerJSON, rest.FlatSwaggerJSON)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to load swagger spec")
	}

	// Create API and server
	api := operations.NewWeaviateAPI(swaggerSpec)
	server := rest.NewServer(api)

	// Configure command line parser
	parser := flags.NewParser(server, flags.Default)
	parser.ShortDescription = "Weaviate"
	server.ConfigureFlags()

	// Add command line option groups
	for _, optsGroup := range api.CommandLineOptionsGroups {
		_, err := parser.AddGroup(optsGroup.ShortDescription, optsGroup.LongDescription, optsGroup.Options)
		if err != nil {
			return nil, errors.Wrap(err, "Failed to add flag group")
		}
	}

	// Parse command line arguments
	if _, err := parser.Parse(); err != nil {
		if fe, ok := err.(*flags.Error); ok && fe.Type == flags.ErrHelp {
			return nil, nil
		}
		return nil, err
	}

	// Configure server
	server.EnabledListeners = []string{"http"}
	p, err := strconv.Atoi(port)
	if err != nil {
		return nil, errors.Wrap(err, "Failed to convert port to int")
	}
	server.Port = p

	// Configure API
	server.ConfigureAPI()

	// Start server in goroutine
	go func() {
		if err := server.Serve(); err != nil && err != http.ErrServerClosed {
			fmt.Printf("Weaviate serve error: %v\n", err)
		}
	}()

	// Handle context cancellation
	go func() {
		<-ctx.Done()
		fmt.Println("Context canceled, shutting down Weaviate server")
		_ = server.Shutdown()
	}()

	// Wait for server to become ready
	time.Sleep(100 * time.Millisecond)
	readyURL := fmt.Sprintf("http://localhost:%d/v1/.well-known/ready", p)
	deadline := time.Now().Add(15 * time.Second)
	fmt.Printf("Waiting for Weaviate to become ready at %s\n", readyURL)

	checkCount := 0
	for {
		checkCount++
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("weaviate did not become ready in time on %s", readyURL)
		}

		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, readyURL, nil)
		resp, err := http.DefaultClient.Do(req)

		if err != nil {
			if checkCount <= 5 || checkCount%5 == 0 {
				fmt.Printf("Weaviate readiness check failed (attempt %d): %v\n", checkCount, err)
			}
		} else {
			defer func() {
				if resp != nil && resp.Body != nil {
					resp.Body.Close()
				}
			}()

			if resp.StatusCode == http.StatusOK {
				fmt.Printf("Weaviate server is ready! (elapsed: %v, checks: %d)\n", time.Since(startTime), checkCount)
				return server, nil
			} else {
				if checkCount <= 5 || checkCount%5 == 0 {
					fmt.Printf("Weaviate not ready yet (attempt %d, status: %d)\n", checkCount, resp.StatusCode)
				}
			}
		}

		time.Sleep(200 * time.Millisecond)
	}
}

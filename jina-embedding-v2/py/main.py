import onnxruntime
import numpy as np
from transformers import AutoTokenizer, PretrainedConfig
import time
import sys
import os
import socket
import threading
import json
import signal

# Global variables to hold loaded model and tokenizer
session = None
tokenizer = None
config = None

SERVER_PORT = 8888
MODEL_PATH = '../model/model.onnx'

# Global server state
server_socket = None
shutdown_requested = False

def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray):
    """Apply mean pooling to model outputs."""
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def handle_inference_request(text):
    """Handle inference request and return embeddings."""
    global session, tokenizer, config
    
    if session is None or tokenizer is None:
        return {"error": "Model not loaded"}
    
    try:
        # Start timing from tokenization
        start_time = time.time()
        
        # Tokenize input
        input_text = tokenizer(text, return_tensors='np')
        
        # Prepare inputs for ONNX model
        inputs = {
            'input_ids': input_text['input_ids'],
            'attention_mask': input_text['attention_mask'],
            'token_type_ids': input_text.get('token_type_ids', np.zeros_like(input_text['input_ids']))
        }
        
        # Run model
        outputs = session.run(None, inputs)[0]
        
        # Record total time including tokenization and inference
        total_time = time.time() - start_time
        
        # Apply mean pooling and normalization to the model outputs
        embeddings = mean_pooling(outputs, input_text["attention_mask"])
        embeddings = embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        
        return {
            "embedding": embeddings[0].tolist(),
            "shape": list(embeddings.shape),
            "inference_time": total_time
        }
        
    except Exception as e:
        return {"error": str(e)}

def handle_client(client_socket):
    """Handle client connection."""
    try:
        data = client_socket.recv(4096).decode('utf-8')
        request = json.loads(data)
        
        if request["command"] == "infer":
            result = handle_inference_request(request["text"])
            response = json.dumps(result)
            response_bytes = response.encode('utf-8')
            client_socket.sendall(response_bytes)
        elif request["command"] == "ping":
            client_socket.send(b'{"status": "pong"}')
        elif request["command"] == "shutdown":
            global shutdown_requested
            shutdown_requested = True
            client_socket.send(b'{"status": "shutting down"}')
        else:
            client_socket.send(b'{"error": "Unknown command"}')
            
    except Exception as e:
        error_response = json.dumps({"error": str(e)})
        client_socket.send(error_response.encode('utf-8'))
    finally:
        client_socket.close()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    print(f"\nReceived signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True
    if server_socket:
        server_socket.close()

def start_server():
    """Start the inference server."""
    global server_socket, shutdown_requested
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', SERVER_PORT))
    server_socket.listen(5)
    
    print(f"Server started on port {SERVER_PORT}")
    
    try:
        while not shutdown_requested:
            try:
                server_socket.settimeout(1.0)  # 1 second timeout for accept
                client_socket, addr = server_socket.accept()
                server_socket.settimeout(None)  # Reset timeout
                
                if shutdown_requested:
                    client_socket.close()
                    break
                    
                client_thread = threading.Thread(target=handle_client, args=(client_socket,))
                client_thread.daemon = True
                client_thread.start()
                
            except socket.timeout:
                continue  # Check shutdown_requested flag
            except OSError as e:
                if shutdown_requested:
                    break
                print(f"Socket error: {e}")
                break
                
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("Server shutting down...")
        if server_socket:
            server_socket.close()

def load_model():
    """Load the model and tokenizer."""
    global session, tokenizer, config
    
    try:
        # Load tokenizer and model config
        print("Loading tokenizer and config...")
        tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')
        config = PretrainedConfig.from_pretrained('jinaai/jina-embeddings-v2-base-en')
        
        # Load ONNX session
        print(f"Loading ONNX model from {MODEL_PATH}...")
        session = onnxruntime.InferenceSession(MODEL_PATH)
        
        # Check model inputs
        print("Model inputs: " + str([input.name for input in session.get_inputs()]))
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to load model and start server."""
    load_model()
    start_server()

if __name__ == '__main__':
    main()


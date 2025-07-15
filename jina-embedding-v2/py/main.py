import onnxruntime
import numpy as np
from transformers import AutoTokenizer, PretrainedConfig
import time
import click
import sys
import os
import pickle
import socket
import threading
import json

# Global variables to hold loaded model and tokenizer
session = None
tokenizer = None
config = None

# File paths for persisting loaded model state
MODEL_STATE_FILE = '.model_state.pkl'
SERVER_PORT = 8888

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
            "shape": embeddings.shape,
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
            client_socket.send(b"pong")
        else:
            client_socket.send(b'{"error": "Unknown command"}')
            
    except Exception as e:
        error_response = json.dumps({"error": str(e)})
        client_socket.send(error_response.encode('utf-8'))
    finally:
        client_socket.close()

def start_server():
    """Start the inference server."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', SERVER_PORT))
    server_socket.listen(5)
    
    print(f"Server started on port {SERVER_PORT}")
    
    try:
        while True:
            client_socket, addr = server_socket.accept()
            client_thread = threading.Thread(target=handle_client, args=(client_socket,))
            client_thread.daemon = True
            client_thread.start()
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        server_socket.close()

@click.group()
def cli():
    """Jina Embeddings v2 ONNX CLI."""
    pass

@cli.command()
@click.option('--model-path', default='../model/model.onnx', help='Path to ONNX model file')
def load(model_path):
    """Load the model and tokenizer."""
    global session, tokenizer, config
    
    try:
        # Load tokenizer and model config
        click.echo("Loading tokenizer and config...")
        tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')
        config = PretrainedConfig.from_pretrained('jinaai/jina-embeddings-v2-base-en')
        
        # Load ONNX session
        click.echo(f"Loading ONNX model from {model_path}...")
        session = onnxruntime.InferenceSession(model_path)
        
        # Check model inputs
        click.echo("Model inputs: " + str([input.name for input in session.get_inputs()]))
        
        # Save model state to file for persistence across processes
        model_state = {
            'model_path': model_path,
            'tokenizer_name': 'jinaai/jina-embeddings-v2-base-en'
        }
        with open(MODEL_STATE_FILE, 'wb') as f:
            pickle.dump(model_state, f)
        
        click.echo("Model loaded successfully!")
        
    except Exception as e:
        click.echo(f"Error loading model: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--model-path', default='../model/model.onnx', help='Path to ONNX model file')
def server(model_path):
    """Start the inference server with model preloaded."""
    global session, tokenizer, config
    
    try:
        # Load tokenizer and model config
        click.echo("Loading tokenizer and config...")
        tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')
        config = PretrainedConfig.from_pretrained('jinaai/jina-embeddings-v2-base-en')
        
        # Load ONNX session
        click.echo(f"Loading ONNX model from {model_path}...")
        session = onnxruntime.InferenceSession(model_path)
        
        # Check model inputs
        click.echo("Model inputs: " + str([input.name for input in session.get_inputs()]))
        click.echo("Model loaded successfully!")
        
        # Start server
        start_server()
        
    except Exception as e:
        click.echo(f"Error loading model: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('text')
def infer(text):
    """Run inference on the provided text."""
    global session, tokenizer, config
    
    # Try to load model from persistent state if not already loaded
    if session is None or tokenizer is None:
        if os.path.exists(MODEL_STATE_FILE):
            try:
                with open(MODEL_STATE_FILE, 'rb') as f:
                    model_state = pickle.load(f)
                
                # Load tokenizer and model config
                tokenizer = AutoTokenizer.from_pretrained(model_state['tokenizer_name'])
                config = PretrainedConfig.from_pretrained(model_state['tokenizer_name'])
                
                # Load ONNX session
                session = onnxruntime.InferenceSession(model_state['model_path'])
                
            except Exception as e:
                click.echo(f"Error loading persisted model: {e}", err=True)
                click.echo("Please run 'load' command first.", err=True)
                sys.exit(1)
        else:
            click.echo("Error: Model not loaded. Please run 'load' command first.", err=True)
            sys.exit(1)
    
    try:
        click.echo(f"Input: {text}")
        
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
        click.echo(f"Inference time: {total_time:.4f} seconds")
        
        # Apply mean pooling and normalization to the model outputs
        embeddings = mean_pooling(outputs, input_text["attention_mask"])
        embeddings = embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        
        click.echo(f"Embedding shape: {embeddings.shape}")
        click.echo(f"First 10 values: {embeddings[0][:10]}")
        
    except Exception as e:
        click.echo(f"Error during inference: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('text')
def client(text):
    """Send inference request to running server."""
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', SERVER_PORT))
        
        request = json.dumps({"command": "infer", "text": text})
        client_socket.send(request.encode('utf-8'))
        
        response = client_socket.recv(4096).decode('utf-8')
        result = json.loads(response)
        
        if "error" in result:
            click.echo(f"Error: {result['error']}", err=True)
        else:
            click.echo(f"Input: {text}")
            click.echo(f"Inference time: {result['inference_time']:.4f} seconds")
            click.echo(f"Embedding shape: {result['shape']}")
            click.echo(f"First 10 values: {result['embedding'][:10]}")
            
        client_socket.close()
        
    except Exception as e:
        click.echo(f"Error connecting to server: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()


#!/usr/bin/env python3
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer

def load_model_and_tokenizer():
    """Load ONNX model and tokenizer"""
    model_dir = Path("../model")
    
    # Load ONNX model
    onnx_path = model_dir / "onnx" / "model.onnx"
    session = ort.InferenceSession(str(onnx_path))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return session, tokenizer

def generate_text(session, tokenizer, prompt, max_length=50):
    """Generate text using the ONNX model"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="np", padding=False, add_special_tokens=False)
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    
    
    batch_size, seq_length = input_ids.shape
    
    # Initialize past key values (16 layers x 2 for key/value)
    past_key_values = {}
    num_layers = 16
    num_key_value_heads = 8  # GQA: num_key_value_heads != num_attention_heads
    head_dim = 64
    
    for i in range(num_layers):
        past_key_values[f"past_key_values.{i}.key"] = np.zeros((batch_size, num_key_value_heads, 0, head_dim), dtype=np.float32)
        past_key_values[f"past_key_values.{i}.value"] = np.zeros((batch_size, num_key_value_heads, 0, head_dim), dtype=np.float32)
    
    # Position IDs
    position_ids = np.arange(seq_length, dtype=np.int64).reshape(1, -1)
    
    generated_ids = input_ids.copy()
    
    for step in range(max_length - seq_length):
        # Prepare inputs for ONNX model
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }
        ort_inputs.update(past_key_values)
        
        # Run inference
        outputs = session.run(None, ort_inputs)
        
        # Get logits and update past key values
        logits = outputs[0]
        
        # Update past key values for next iteration
        for i in range(num_layers):
            past_key_values[f"past_key_values.{i}.key"] = outputs[1 + i * 2]
            past_key_values[f"past_key_values.{i}.value"] = outputs[2 + i * 2]
        
        # Get next token (greedy decoding)
        next_token_id = np.argmax(logits[0, -1, :])
        
        # Stop if EOS token is generated
        if next_token_id == tokenizer.eos_token_id:
            break
        
        # Append to generated sequence
        generated_ids = np.concatenate([
            generated_ids, 
            np.array([[next_token_id]], dtype=np.int64)
        ], axis=1)
        
        # For next iteration, input_ids is just the new token
        input_ids = np.array([[next_token_id]], dtype=np.int64)
        
        # Update attention mask
        attention_mask = np.concatenate([
            attention_mask,
            np.ones((1, 1), dtype=np.int64)
        ], axis=1)
        
        # Update position_ids for next token
        position_ids = np.array([[attention_mask.shape[1] - 1]], dtype=np.int64)
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    print("Loading ONNX model and tokenizer...")
    session, tokenizer = load_model_and_tokenizer()
    
    print(f"Model inputs: {[inp.name for inp in session.get_inputs()]}")
    print(f"Model outputs: {[out.name for out in session.get_outputs()]}")
    
    # Test with a simple prompt
    prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: {prompt}")
    
    generated = generate_text(session, tokenizer, prompt, max_length=30)
    print(f"Generated: {generated}")

if __name__ == "__main__":
    main()
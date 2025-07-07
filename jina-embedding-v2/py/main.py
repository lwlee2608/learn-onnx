import onnxruntime
import numpy as np
from transformers import AutoTokenizer, PretrainedConfig
import time

# Mean pool function
def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray):
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

# Load tokenizer and model config
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')
config = PretrainedConfig.from_pretrained('jinaai/jina-embeddings-v2-base-en')

# Load ONNX session
model_path = '../model/model.onnx'
session = onnxruntime.InferenceSession(model_path)

# Check model inputs
print("Model inputs:", [input.name for input in session.get_inputs()])

# Input text
input_text_str = "This is an apple"
# input_text_str = "On the morning of April 16, 2024, I attended the annual AI Innovation Conference in downtown San Francisco. The keynote speaker, Dr. Evelyn Chen, discussed the ethical implications of autonomous decision-making systems in healthcare. I remember the room was filled with experts from various fields, including data science, medicine, and law. After her talk, I had a conversation with a software engineer named Miguel who was developing a diagnostic tool powered by GPT-4. He shared insights about real-world challenges in gathering unbiased medical data. Later, I participated in a roundtable about data privacy and shared my perspective on how granular access controls could help protect sensitive patient information. The day ended with a networking session where I met professionals interested in AI governance. This experience gave me new insights into balancing innovation and ethics."
print(f"Input: {input_text_str}")

# Start timing from tokenization
start_time = time.time()

# Tokenize input
input_text = tokenizer(input_text_str, return_tensors='np')

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
print(f"Total time (tokenization + inference): {total_time:.4f} seconds")

print(outputs)

# Apply mean pooling and normalization to the model outputs
embeddings = mean_pooling(outputs, input_text["attention_mask"])
embeddings = embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)

print(embeddings.shape)  # Output shape
print("First 10 values:", embeddings[0][:10])


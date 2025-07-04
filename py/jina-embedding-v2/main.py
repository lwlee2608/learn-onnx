import onnxruntime
import numpy as np
from transformers import AutoTokenizer, PretrainedConfig

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

# Tokenize input
input_text = tokenizer('This is an apple', return_tensors='np')

# ONNX session
model_path = 'model/model.onnx'
session = onnxruntime.InferenceSession(model_path)

# Check model inputs
print("Model inputs:", [input.name for input in session.get_inputs()])

# Prepare inputs for ONNX model
inputs = {
    'input_ids': input_text['input_ids'],
    'attention_mask': input_text['attention_mask'],
    'token_type_ids': input_text.get('token_type_ids', np.zeros_like(input_text['input_ids']))
}

# Run model
outputs = session.run(None, inputs)[0]

print(outputs)

# Apply mean pooling and normalization to the model outputs
embeddings = mean_pooling(outputs, input_text["attention_mask"])
embeddings = embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)

print(embeddings.shape)  # Output shape
print("First 10 values:", embeddings[0][:10])


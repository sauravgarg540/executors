# Transformer Torch Encoder

**Transformer Torch Encoder** is a class that encodes sentences into embeddings.

The following parameters can be used:

- `pretrained_model_name_or_path` (str, default 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens'): Path to pretrained model or name of the model in transformers package
- `base_tokenizer_model` (str, optional): Base tokenizer model
- `pooling_strategy` (str, default 'mean'): Pooling Strategy
- `layer_index` (int, default -1): Index of the layer which contains the embeddings
- `max_length` (int, optional): Max length argument for the tokenizer
- `embedding_fn_name` (str, default __call__): Function to call on the model in order to get output
- `device` (str, default 'cpu'): Device to be used. Use 'cuda' for GPU
- `default_traversal_paths` (List[str], Optional): Used in the encode method an define traversal on the received `DocumentArray`. Defaults to ['r']
- `default_batch_size` (int, default 32): Defines the batch size for inference on the loaded PyTorch model.





## Usage


Use the prebuilt images from JinaHub in your Python code: 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://TransformerTorchEncoder',
               volumes='/your_user/.cache/huggingface:/root/.cache/huggingface')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://TransformerTorchEncoder'
    volumes: '/your_user/.cache/huggingface:/root/.cache/huggingface'
    uses_with: 
      target_size: 42
``` 
This does not support GPU at the moment.


#### GPU usage

You can use the GPU via the source code. Therefore, you need a matching CUDA version
and GPU drivers installed on your system. 
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://TransformerTorchEncoder'
    uses_with:
      device: 'cuda'
```
Alternatively, use the jinahub gpu docker container. Therefore, you need GPU
drivers installed on your system and nvidia-docker installed.
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://TransformerTorchEncoder/gpu'
    gpus: all
    uses_with:
      device: 'cuda'
```


## Usage

```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://TransformerTorchEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(), return_results=True)
	print(f'{resp}')
```

### Inputs 

`Document` with text in the `text` field.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` (model-dependent) with `dtype=nfloat32`.


## Reference
- Available models: https://huggingface.co/transformers/pretrained_models.html
- More available models: https://huggingface.co/sentence-transformers

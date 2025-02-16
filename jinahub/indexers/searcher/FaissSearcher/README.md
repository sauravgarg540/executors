# FaissSearcher

**FaissSearcher** is a Faiss-powered vector Searcher.

Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. Some of the most useful algorithms are implemented on the GPU. It is developed by Facebook AI Research.






## Usage

Check [tests](tests) for an example on how to use it.

### Loading data

Since this is a "Searcher"-type Executor, it does not _index_ new data. Rather they are write-once classes, which take as data source a `dump_path`. 

This can be provided in different ways:

- in the YAML definition
  
```yaml
jtype: FaissSearcher
with:
    dump_path: /tmp/your_dump_location
...
```

- from the `Flow.rolling_update` method. See [README](../../../../README.md).

The folder needs to contain the data exported from your Indexer. Again, see [README](../../../../README.md). 


```python
import numpy as np
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://FaissSearcher')

with f:
    resp = f.post(on='/search', inputs=Document(embedding=np.array([1,2,3])), return_results=True)
    print(f'{resp}')
```

### Training

To use a trainable Faiss indexer (e.g., _IVF_, _PQ_ based), we can first train the indexer with the data from `train_data_file`:

```python
from jina import Flow
import numpy as np

train_data_file = 'train.npy'
train_data = np.array(np.random.random([10240, 256]), dtype=np.float32)
np.save(train_data_file, train_data)

f = Flow().add(
    uses="jinahub://FaissSearcher",
    timeout_ready=-1,
    uses_with={
      'index_key': 'IVF10_HNSW32,PQ64',
      'trained_index_file': 'faiss.index',
    },
)

with f:
    # the trained index will be dumped to "faiss.index"
    f.post(on='/train', parameters={'train_data_file': train_data_file})
```

Then, we can directly use the trained indexer with providing `trained_index_file`:

```python
f = Flow().add(
    uses="jinahub://FaissSearcher",
    timeout_ready=-1,
    uses_with={
      'index_key': 'IVF10_HNSW32,PQ64',
      'trained_index_file': 'faiss.index',
      'dump_path': '/path/to/dump_file'
    },
)
```

### Inputs 

`Document` with `.embedding` the same shape as the `Documents` it has stored.

### Returns

Attaches matches to the Documents sent as inputs, with the id of the match, and its embedding. For retrieving the full metadata (original text or image blob), use a [key-value searcher](./../../keyvalue).


## Reference

- [Facebook Research's Faiss](https://github.com/facebookresearch/faiss)

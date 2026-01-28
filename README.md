# GaussDB for LangChain

An integration package connecting GaussDB and LangChain, supports quick connection to GaussDB and integrates LangChain workflows.

## Installation

- git clone this repo, then install with:

```shell
poetry install
```

- install with pip (already been released):
```shell
pip install langchain_gaussdb-0.1.0-py3-none-any.whl
```

## Quick Start

### Basic Usage

```python
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_gaussdb import GaussVectorSettings, GaussVectorStore, IvfFlatParams

embeddings = OpenAIEmbeddings(
    model="Qwen/Qwen3-Embedding-8B",
    api_key="xxx",
    base_url="https://api.siliconflow.cn/v1",
    dimensions=1024
)
config = GaussVectorSettings(
    host="10.25.106.120",
    port=9800,
    user="langchain_gv",
    password="xxx",
    database="postgres",
    table_name="my_docs"
)

vector_store = GaussVectorStore(
    embedding=embeddings,
    config=config
)

# Insert documents
docs = [
    Document(page_content="Quantum computing basics", metadata={"field": "physics"}),
    Document(page_content="Neural network architectures", metadata={"field": "ai"})
]
vector_store.add_documents(docs)

# Semantic search
results = vector_store.similarity_search("deep learning models", k=1)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")
```

## Configuration Guide

### Connection Settings

| Parameter             | Default                 | Description                                                                                       |
|-----------------------|-------------------------|---------------------------------------------------------------------------------------------------|
| `host`                | localhost               | Database server address                                                                           |
| `port`                | 5432                    | Database connection port                                                                          |
| `user`                | gaussdb                 | Database username                                                                                 |
| `password`            | -                       | Complex password string                                                                           |
| `database`            | postgres                | Default database name                                                                             |
| `min_connections`     | 1                       | Connection pool minimum size                                                                      |
| `max_connections`     | 5                       | Connection pool maximum size                                                                      |
| `table_name`          | langchain_docs          | Name of the table for storing vector data and metadata                                            |
| `index_type`          | IndexType.GSDISKANN     | Vector index algorithm type. Options: GsIVFFLAT or GsDiskANN\nDefault is GsDiskANN.               |
| `index_params`        | DiskAnnParams()         | Vector index params, different index_type require different index_params.                         |
| `vector_type`         | VectorType.float_vector | Type of vector representation to use. Options: floatvector or boolvector\nDefault is floatvector. |
| `distance_strategy`   | DistanceStrategy.COSINE | Vector similarity metric to use for retrieval. Options: l2, cosine, hamming.\n Default is cosine. |
| `embedding_dimension` | 1024                    | Dimensionality of the vector embeddings.                                                          |

#### Supported Combinations

| Index Types | Dimensions | Vector Type            | Supported Distance Strategies |
|-------------|-----------|------------------------|-------------------------------|
| GsIVFFLAT   | <=1024    | floatvector/boolvector | l2/cosine/hamming             |
| GsDiskANN   | <=4096    | floatvector            | l2/cosine                     |

## Advanced Usage

### Hybrid Search with Metadata

```python
# Filter by metadata with vector search
results = vector_store.similarity_search(
    query="machine learning",
    k=3,
    filter={"publish_year": 2023, "category": "research"},
)

# Perform similarity search and receive corresponding scores
results_with_score = vectorstore.similarity_search_with_score(
    query="test query",
    k=2
)
```

## API Reference

### Core Methods
| Method                         | Description                                   |
|--------------------------------|-----------------------------------------------|
| `add_documents`                | Insert documents with automatic embedding     |
| `similarity_search `           | Basic vector similarity search                |
| `similarity_search_with_score` | Return (document, similarity_score) tuples   |
| `delete`                       | Remove documents by ID list                  |
| `drop_table`                   | Delete entire collection                     |

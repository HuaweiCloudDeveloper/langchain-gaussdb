from enum import Enum
from typing import Union

from pydantic import BaseModel, Field


class IndexType(str, Enum):
    """Supported index types for vector search"""
    GSIVFFLAT = "GsIVFFLAT"  # support floatvector and boolvector type, vector max dimension is 1024
    GSDISKANN = "GsDiskANN"  # support floatvector type, vector max dimension is 4096


class VectorType(str, Enum):
    """Supported vector data types in GaussVector"""
    float_vector = "floatvector"
    bool_vector = "boolvector"


class DistanceStrategy(str, Enum):
    """Supported distance metrics for vector similarity search"""
    EUCLIDEAN = "l2"  # support floatvector type
    COSINE = "cosine"  # support floatvector type
    HAMMING = "hamming"  # support boolvector type


class IvfFlatParams(BaseModel):
    """GSIVFFLAT index parameters"""
    ivf_nlist: int = 100
    ivf_nlist2: int = 0
    num_parallels: int = 1
    enable_quantization: bool = False
    quantization_type: str = 'pq'


class DiskAnnParams(BaseModel):
    """GsDiskANN index parameters"""
    pq_nseg: int = 1
    pq_nclus: int = 16
    num_parallels: int = 10
    enable_pq: bool = True
    quantization_type: str = 'pq'


class GaussVectorSettings(BaseModel):
    """Configuration settings for GaussVector database connection and vector search capabilities"""
    # Database connection settings
    host: str = "localhost"
    port: int = 5432
    user: str = "gaussdb"
    password: str = "Test@123456"
    database: str = "postgres"
    min_connections: int = 1
    max_connections: int = 5
    table_name: str = "langchain_docs"

    # Vector index settings
    index_type: IndexType = IndexType.GSDISKANN
    index_params: Union[IvfFlatParams, DiskAnnParams] = Field(default_factory=DiskAnnParams)
    vector_type: VectorType = VectorType.float_vector
    distance_strategy: DistanceStrategy = DistanceStrategy.COSINE
    embedding_dimension: int = 1024

    @property
    def index_operator(self) -> str:
        strategy_to_op = {
            DistanceStrategy.EUCLIDEAN: "<->",
            DistanceStrategy.COSINE: "<+>",
            DistanceStrategy.HAMMING: "<#>",
        }
        return strategy_to_op[self.distance_strategy]

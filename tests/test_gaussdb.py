import pytest
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_gaussdb import GaussVectorSettings, GaussVectorStore, IvfFlatParams


class TestGaussVectorStore:
    """Integration tests for GaussVectorStore"""

    @pytest.fixture(scope="module")
    def vectorstore(self):
        """Create a vectorstore for integration tests"""
        embeddings = OpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-8B",
            api_key="sk-hakqqisicskmfuhnrziueviywygkbjbkmgcqtvchlcmjtnpl",
            base_url="https://api.siliconflow.cn/v1",
            dimensions=1024
        )
        config = GaussVectorSettings(
            host="10.25.106.120",
            port=9800,
            user="fusion_ai",
            password="Gauss_234",
            database="postgres",
            table_name="test_vector"
        )

        vector_store = GaussVectorStore(
            embedding=embeddings,
            config=config
        )
        try:
            yield vector_store
        finally:
            vector_store.drop_table()

    def test_empty_search(self, vectorstore):
        """Test search behavior with empty vectorstore"""
        # Search in empty vectorstore
        results = vectorstore.similarity_search("test", k=5)
        assert len(results) == 0

        results_with_score = vectorstore.similarity_search_with_score("test", k=5)
        assert len(results_with_score) == 0

    def test_add_and_search(self, vectorstore):
        """Test basic document addition and search functionality"""
        documents = [
            Document(page_content="Hello world", metadata={"source": "test1"}),
            Document(page_content="Python programming", metadata={"source": "test2"}),
            Document(page_content="Machine learning", metadata={"source": "test3"}),
        ]

        # 1. Add documents
        ids = vectorstore.add_documents(documents)
        assert len(ids) == 3

        # 2. Test that we can retrieve all documents by searching with empty string
        all_results = vectorstore.similarity_search("", k=10)
        assert len(all_results) >= 3

        content_list = [doc.page_content for doc in all_results]
        assert "Hello world" in content_list
        assert "Python programming" in content_list
        assert "Machine learning" in content_list

        # 3. Test similarity search with filter
        results_filter = vectorstore.similarity_search("test query", k=3, filter={"source": "test2"})
        assert len(results_filter) >= 1

        assert all(
            doc.metadata.get("source") == "test2"
            for doc in results_filter
        )

        # 4. Test similarity search with score
        results_with_score = vectorstore.similarity_search_with_score("test query", k=2)
        assert len(results_with_score) >= 1
        assert all(
            isinstance(result, tuple) and len(result) == 2
            for result in results_with_score
        )

    def test_get_and_delete(self, vectorstore):
        """Test document retrieval and deletion functionality"""
        documents = [
            Document(page_content="Get by IDs test document 1", metadata={"id": "1"}),
            Document(page_content="Get by IDs test document 2", metadata={"id": "2"}),
            Document(page_content="Get by IDs test document 3", metadata={"id": "3"}),
        ]

        ids = vectorstore.add_documents(documents)

        # 1. Retrieve by IDs
        retrieved_docs = vectorstore.get_by_ids(ids[:2])
        assert len(retrieved_docs) == 2

        # 2. Check content
        content_list = [doc.page_content for doc in retrieved_docs]
        assert "Get by IDs test document 1" in content_list
        assert "Get by IDs test document 2" in content_list

        # 3. Test retrieving all documents
        all_retrieved_docs = vectorstore.get_by_ids(ids)
        assert len(all_retrieved_docs) == 3
        all_content_list = [doc.page_content for doc in all_retrieved_docs]
        assert "Get by IDs test document 1" in all_content_list
        assert "Get by IDs test document 2" in all_content_list
        assert "Get by IDs test document 3" in all_content_list

        # 4. Delete by IDs
        vectorstore.delete(ids[1:])
        assert len(vectorstore.get_by_ids([ids[0]])) == 1
        assert len(vectorstore.get_by_ids([ids[1]])) == 0

        vectorstore.delete()
        assert len(vectorstore.get_by_ids([ids[0]])) == 0
        assert len(vectorstore.get_by_ids(ids)) == 0

    def test_from_documents(self, vectorstore):
        """Test creating vectorstore from documents"""
        documents = [
            Document(page_content="Integration test 1", metadata={"source": "int1"}),
            Document(page_content="Integration test 2", metadata={"source": "int2"}),
            Document(page_content="Integration test 3", metadata={"source": "int3"}),
        ]

        new_vectorstore = GaussVectorStore.from_documents(
            documents=documents,
            embedding=vectorstore.embedding_function,
            config=vectorstore.config
        )

        # 1. Verify all texts are present by searching with empty string
        all_results = new_vectorstore.similarity_search("", k=10)
        content_list = [doc.page_content for doc in all_results]
        for document in documents:
            assert document.page_content in content_list

        # Test that search returns some results
        results = new_vectorstore.similarity_search("test query", k=2)
        assert len(results) >= 1

    def test_different_index_types(self, vectorstore):
        """Test different index types"""
        documents = [
            Document(page_content="GsIVFFLAT test 1", metadata={"source": "test1"}),
            Document(page_content="GsIVFFLAT test 2", metadata={"source": "test2"}),
            Document(page_content="GsIVFFLAT test 3", metadata={"source": "test3"}),
        ]
        config = GaussVectorSettings(
            host="10.25.106.120",
            port=9800,
            user="fusion_ai",
            password="Gauss_234",
            database="postgres",
            table_name="test_vector2",
            index_type="GsIVFFLAT",
            index_params=IvfFlatParams(ivf_nlist=128, num_parallels=32, enable_quantization=True),
            distance_strategy="l2"
        )
        vectorstore2 = GaussVectorStore(
            embedding=vectorstore.embedding_function,
            config=config
        )

        ids = vectorstore2.add_documents(documents)
        assert len(ids) == 3

        results = vectorstore2.similarity_search("test query", k=2)
        assert len(results) >= 1
        assert all(
            doc.page_content in ["GsIVFFLAT test 1", "GsIVFFLAT test 2", "GsIVFFLAT test 3"]
            for doc in results
        )

        vectorstore2.drop_table()

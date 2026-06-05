from unittest.mock import patch, MagicMock
import pytest
from langchain_core.documents import Document
from src.tools import _get_reranker, search_steuerbuch, create_local_search_tool


# ==============================================================================
# ADVANCED INTEGRATION TESTS FOR PIPELINES & FACTORIES
# ==============================================================================

# ---- 1. Testing Lazy Loading (Singleton Pattern) ----

def test_get_reranker_lazy_loading_and_caching():
    """Validates that the heavy CrossEncoder is only initialized once (Singleton)."""
    with patch("src.tools._reranker", None), \
            patch("src.tools.CrossEncoder") as mock_cross_encoder_class:
        # Act: First call should trigger instantiation
        instance_1 = _get_reranker()
        assert mock_cross_encoder_class.call_count == 1

        # Act: Second call should return the cached instance without calling the constructor again
        with patch("src.tools._reranker", instance_1):
            instance_2 = _get_reranker()
            assert mock_cross_encoder_class.call_count == 1
            assert instance_1 == instance_2


# ---- 2. Testing Fuzzy Matching, Sorting, and URL Capping ----

def test_search_steuerbuch_fuzzy_scoring_sorting_and_capping():
    """Ensures results are fuzzy-ranked by title and capped strictly at MAX_URL_READS."""
    fake_api_response = MagicMock()
    # Simulate an API returning 4 results. The 3rd one (index 2) has the perfect match title.
    fake_api_response.json.return_value = {
        "numberOfResults": 4,
        "data": [
            {"recordfield-flexdataschemainputftitel": "Generelle Infos", "link": "/link1",
             "recordfield-flexdataschemainputfzstb-nummer": "1"},
            {"recordfield-flexdataschemainputftitel": "Andere Steuern", "link": "/link2",
             "recordfield-flexdataschemainputfzstb-nummer": "2"},
            {"recordfield-flexdataschemainputftitel": "Miete und Eigenmietwert", "link": "/link3",
             "recordfield-flexdataschemainputfzstb-nummer": "3"},  # Best match for 'Miete'
            {"recordfield-flexdataschemainputftitel": "Infrastruktur", "link": "/link4",
             "recordfield-flexdataschemainputfzstb-nummer": "4"}
        ]
    }

    fake_page_response = MagicMock()
    fake_page_response.text = "<html><body><main>Content</main></body></html>"

    with patch("src.tools.requests.get",
               side_effect=[fake_api_response, fake_page_response, fake_page_response, fake_page_response]):
        result = search_steuerbuch.invoke({"query": "Miete"})

    # Assertions:
    # 1. Check that the compiled entries do not exceed MAX_URL_READS (3)
    assert "3 entries compiled" in result
    # 2. Check that the best fuzzy match (ZStB Nr. 3) was prioritized and crawled first
    assert "[ZStB Nr. 3 – Miete und Eigenmietwert]" in result


# ---- 3. Testing Factory Closure, Deduplication, and Reranking ----

def test_local_search_tool_deduplication_and_reranking_pipeline():
    """Tests the factory closure tool to ensure chunks are deduplicated and limited to TOP_K."""

    # Setup: Mock the Vector Store
    fake_vector_store = MagicMock()

    # Create two duplicate documents that share the exact same content and metadata
    doc_a = Document(page_content="Text content about wealth tax.", metadata={"source": "law.pdf", "page": 1})
    doc_b = Document(page_content="Text content about wealth tax.",
                     metadata={"source": "law.pdf", "page": 1})  # Duplicate
    doc_c = Document(page_content="Distinct text about deductions.", metadata={"source": "law.pdf", "page": 2})

    # Vector store returns these documents for any query invocation
    fake_vector_store.similarity_search.return_value = [doc_a, doc_b, doc_c]

    # Setup: Mock the internal LLM query expansion to return fixed queries
    with patch("src.tools.generate_queries", return_value=["Alt Q1", "Alt Q2", "Alt Q3"]), \
            patch("src.tools._get_reranker") as mock_reranker_fn:
        # Mock the reranker model behavior to output a score sequence matching the flattened unique documents
        mock_model = MagicMock()
        # We expect 2 unique documents after our internal composite-key deduplication engine fires
        mock_model.predict.return_value = [0.95, 0.40]
        mock_reranker_fn.return_value = mock_model

        # Act: Initialize the factory closure tool
        local_search_tool = create_local_search_tool(fake_vector_store)
        result = local_search_tool.invoke({"query": "Vermögenssteuer"})

    # Assertions:
    # 1. Verify that the unique chunk content is rendered inside the final response string
    assert "Text content about wealth tax." in result
    # 2. Verify that despite duplicate hits across multi-queries, the unique separator only occurs once
    assert result.count("Source: law.pdf, Page 1") == 1
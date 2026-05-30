from unittest.mock import patch, MagicMock
import pytest
from src.tools import generate_queries, _fetch_page_text, search_steuerbuch

# ---- generate_queries ----

def test_generate_queries_returns_n_queries():
    fake_result = MagicMock()
    fake_result.content = "Query1\nQuery2\nQuery3"

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_result

    with patch("src.tools._query_llm", fake_llm):
        result = generate_queries("Homeoffice Abzug", n=3)

    assert len(result) == 3

def test_generate_queries_filters_empty_lines():
    fake_result = MagicMock()
    fake_result.content = "Query1\n\nQuery2\n"

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_result

    with patch("src.tools._query_llm", fake_llm):
        result = generate_queries("Homeoffice Abzug", n=3)

    assert "" not in result

# ---- _fetch_page_text ----

def test_fetch_page_text_extracts_content():
    fake_response = MagicMock()
    fake_response.text = """
        <html><body>
            <div class="mod-smaller-headings"><p>Steuerinhalt</p></div>
            <nav>Navigation</nav>
        </body></html>
    """

    with patch("src.tools.requests.get", return_value=fake_response):
        result = _fetch_page_text("https://zh.ch/steuer")

    assert "Steuerinhalt" in result
    assert "Navigation" not in result

def test_fetch_page_text_fallback_to_main():
    fake_response = MagicMock()
    fake_response.text = """
        <html><body>
            <main><p>Hauptinhalt</p></main>
        </body></html>
    """

    with patch("src.tools.requests.get", return_value=fake_response):
        result = _fetch_page_text("https://zh.ch/steuer")

    assert "Hauptinhalt" in result

# ---- search_steuerbuch ----

def test_search_steuerbuch_returns_results():
    fake_api_response = MagicMock()
    fake_api_response.json.return_value = {
        "numberOfResults": 1,
        "data": [{
            "recordfield-flexdataschemainputftitel": "Homeoffice Abzug",
            "recordfield-flexdataschemainputfzstb-nummer": "17.1",
            "link": "/de/steuern/homeoffice",
            "_score": 90
        }]
    }

    fake_page_response = MagicMock()
    fake_page_response.text = "<html><body><main><p>Homeoffice Regelung</p></main></body></html>"

    with patch("src.tools.requests.get", side_effect=[fake_api_response, fake_page_response]):
        result = search_steuerbuch.invoke({"query": "Homeoffice"})

    assert "Homeoffice" in result
    assert "zh.ch" in result

def test_search_steuerbuch_no_results():
    fake_api_response = MagicMock()
    fake_api_response.json.return_value = {
        "numberOfResults": 0,
        "data": []
    }

    with patch("src.tools.requests.get", return_value=fake_api_response):
        result = search_steuerbuch.invoke({"query": "Unbekannt"})

    assert "Keine Steuerbuch-Einträge" in result
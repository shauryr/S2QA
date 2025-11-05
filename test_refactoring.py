#!/usr/bin/env python3
"""
Test script to verify the refactoring changes.

This script tests:
1. Core module imports
2. Individual module functionality
3. Integration with main application files
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all core modules can be imported."""
    print("=" * 60)
    print("Testing Core Module Imports")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    # Test individual module imports
    modules_to_test = [
        ("core.citations", ["get_citation", "extract_numbers_in_brackets", "generate_used_reference_display"]),
        ("core.ui_helpers", ["get_twitter_badge", "get_github_badge", "documents_to_df"]),
        ("core.prompts", ["generate_prompt", "create_context_chatgpt"]),
        ("core.embeddings", ["get_specter_embeddings"]),
        ("core.search", ["search_papers", "preprocess_query", "rerank_papers"]),
        ("core.pdf_handler", ["PDFHandler"]),
        ("core.data_loader", ["SemanticScholarReader"]),
    ]

    for module_name, functions in modules_to_test:
        try:
            module = __import__(module_name, fromlist=functions)
            for func_name in functions:
                if hasattr(module, func_name):
                    print(f"✓ {module_name}.{func_name}")
                    tests_passed += 1
                else:
                    print(f"✗ {module_name}.{func_name} - NOT FOUND")
                    tests_failed += 1
        except Exception as e:
            print(f"✗ {module_name} - IMPORT FAILED: {e}")
            tests_failed += len(functions)

    print(f"\nImport Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_citations():
    """Test citation module functions."""
    print("\n" + "=" * 60)
    print("Testing Citations Module")
    print("=" * 60)

    try:
        from core.citations import get_citation, extract_numbers_in_brackets

        # Test citation generation
        metadata = {
            "title": "Deep Learning for NLP",
            "authors": ["John Doe", "Jane Smith"],
            "year": 2023,
            "venue": "ACL"
        }
        citation = get_citation(metadata)
        assert "Deep Learning for NLP" in citation
        assert "2023" in str(citation)
        print("✓ get_citation() works correctly")

        # Test citation extraction
        text = "This is supported by research [1] and [2] and also [5]."
        numbers = extract_numbers_in_brackets(text)
        assert numbers == [1, 2, 5]
        print("✓ extract_numbers_in_brackets() works correctly")

        return True
    except Exception as e:
        print(f"✗ Citations module test failed: {e}")
        return False


def test_ui_helpers():
    """Test UI helpers module functions."""
    print("\n" + "=" * 60)
    print("Testing UI Helpers Module")
    print("=" * 60)

    try:
        from core.ui_helpers import get_twitter_badge, get_github_badge, display_questions

        # Test badge generation
        twitter_badge = get_twitter_badge()
        assert "twitter" in twitter_badge.lower()
        print("✓ get_twitter_badge() works correctly")

        github_badge = get_github_badge()
        assert "github" in github_badge.lower()
        print("✓ get_github_badge() works correctly")

        # Test questions display
        questions = ["Q1?", "Q2?", "Q3?"]
        display = display_questions(questions)
        assert "Q1?" in display and "Q2?" in display
        print("✓ display_questions() works correctly")

        return True
    except Exception as e:
        print(f"✗ UI helpers module test failed: {e}")
        return False


def test_prompts():
    """Test prompts module functions."""
    print("\n" + "=" * 60)
    print("Testing Prompts Module")
    print("=" * 60)

    try:
        from core.prompts import create_context_chatgpt
        import pandas as pd

        # Create sample dataframe
        df = pd.DataFrame({
            'tldr': ['Summary 1', 'Summary 2', 'Summary 3'],
            'paperId': ['abc123', 'def456', 'ghi789']
        })

        context = create_context_chatgpt("test question", df, k=2)
        assert "Summary 1" in context
        assert "Summary 2" in context
        assert "abc123" in context
        print("✓ create_context_chatgpt() works correctly")

        return True
    except Exception as e:
        print(f"✗ Prompts module test failed: {e}")
        return False


def test_search():
    """Test search module functions."""
    print("\n" + "=" * 60)
    print("Testing Search Module")
    print("=" * 60)

    try:
        from core.search import preprocess_query

        # Test query preprocessing
        query = "What is the impact of deep learning on NLP?"
        processed = preprocess_query(query, remove_stopwords=False)
        assert processed == "what is the impact of deep learning on nlp?"
        print("✓ preprocess_query() works correctly")

        return True
    except Exception as e:
        print(f"✗ Search module test failed: {e}")
        return False


def test_pdf_handler():
    """Test PDF handler module."""
    print("\n" + "=" * 60)
    print("Testing PDF Handler Module")
    print("=" * 60)

    try:
        from core.pdf_handler import PDFHandler

        # Test PDFHandler initialization
        handler = PDFHandler(base_dir="test_pdfs")
        assert handler.base_dir == "test_pdfs"
        print("✓ PDFHandler initialization works correctly")

        # Clean up test directory
        if os.path.exists("test_pdfs") and len(os.listdir("test_pdfs")) == 0:
            os.rmdir("test_pdfs")

        return True
    except Exception as e:
        print(f"✗ PDF handler module test failed: {e}")
        return False


def test_main_app_imports():
    """Test that main application files can import from core."""
    print("\n" + "=" * 60)
    print("Testing Main Application Integration")
    print("=" * 60)

    try:
        # Test backend.py imports
        with open('backend.py', 'r') as f:
            content = f.read()
            assert 'from core import SemanticScholarReader' in content
            print("✓ backend.py imports from core correctly")

        # Test S2QA.py imports
        with open('S2QA.py', 'r') as f:
            content = f.read()
            assert 'from core import' in content
            print("✓ S2QA.py imports from core correctly")

        return True
    except Exception as e:
        print(f"✗ Main app integration test failed: {e}")
        return False


def test_legacy_structure():
    """Verify legacy files are properly moved."""
    print("\n" + "=" * 60)
    print("Testing Legacy File Structure")
    print("=" * 60)

    try:
        # Check legacy directory exists
        assert os.path.exists('legacy'), "legacy/ directory not found"
        print("✓ legacy/ directory exists")

        # Check legacy subdirectories
        assert os.path.exists('legacy/streamlit'), "legacy/streamlit/ not found"
        print("✓ legacy/streamlit/ exists")

        assert os.path.exists('legacy/serp_api_qa'), "legacy/serp_api_qa/ not found"
        print("✓ legacy/serp_api_qa/ exists")

        # Check old utils preserved
        assert os.path.exists('utils_old.py'), "utils_old.py not found"
        print("✓ utils_old.py preserved")

        return True
    except Exception as e:
        print(f"✗ Legacy structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("*" * 60)
    print("S2QA REFACTORING TEST SUITE")
    print("*" * 60)

    all_tests = [
        ("Import Tests", test_imports),
        ("Citations Module", test_citations),
        ("UI Helpers Module", test_ui_helpers),
        ("Prompts Module", test_prompts),
        ("Search Module", test_search),
        ("PDF Handler Module", test_pdf_handler),
        ("Main App Integration", test_main_app_imports),
        ("Legacy Structure", test_legacy_structure),
    ]

    results = []
    for test_name, test_func in all_tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} encountered an error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n")
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\n{passed}/{total} test suites passed")

    if passed == total:
        print("\n🎉 All tests passed! Refactoring is successful.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

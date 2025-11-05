#!/usr/bin/env python3
"""
Structure test - verifies refactoring without requiring dependencies.

This test validates:
1. Module structure is correct
2. Files are properly organized
3. Imports are updated
4. Code compiles without syntax errors
"""

import os
import sys
import ast

def test_core_structure():
    """Test that core module has correct structure."""
    print("=" * 60)
    print("Testing Core Module Structure")
    print("=" * 60)

    required_files = [
        'core/__init__.py',
        'core/data_loader.py',
        'core/pdf_handler.py',
        'core/search.py',
        'core/embeddings.py',
        'core/citations.py',
        'core/ui_helpers.py',
        'core/prompts.py',
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False

    return all_exist


def test_syntax():
    """Test that all Python files have valid syntax."""
    print("\n" + "=" * 60)
    print("Testing Python Syntax")
    print("=" * 60)

    files_to_check = [
        'core/__init__.py',
        'core/data_loader.py',
        'core/pdf_handler.py',
        'core/search.py',
        'core/embeddings.py',
        'core/citations.py',
        'core/ui_helpers.py',
        'core/prompts.py',
        'S2QA.py',
        'backend.py',
    ]

    all_valid = True
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                ast.parse(f.read())
            print(f"✓ {file_path} - valid Python syntax")
        except SyntaxError as e:
            print(f"✗ {file_path} - syntax error: {e}")
            all_valid = False

    return all_valid


def test_imports_updated():
    """Test that main files import from core."""
    print("\n" + "=" * 60)
    print("Testing Import Statements")
    print("=" * 60)

    # Check backend.py
    with open('backend.py', 'r') as f:
        backend_content = f.read()

    if 'from core import SemanticScholarReader' in backend_content:
        print("✓ backend.py imports from core")
    else:
        print("✗ backend.py doesn't import from core")
        return False

    # Check S2QA.py
    with open('S2QA.py', 'r') as f:
        s2qa_content = f.read()

    if 'from core import' in s2qa_content:
        print("✓ S2QA.py imports from core")
    else:
        print("✗ S2QA.py doesn't import from core")
        return False

    return True


def test_legacy_preserved():
    """Test that legacy code is preserved."""
    print("\n" + "=" * 60)
    print("Testing Legacy Preservation")
    print("=" * 60)

    checks = [
        ('legacy/', 'legacy directory exists'),
        ('legacy/streamlit/', 'legacy streamlit exists'),
        ('legacy/serp_api_qa/', 'legacy serp_api_qa exists'),
        ('utils_old.py', 'original utils preserved'),
    ]

    all_present = True
    for path, desc in checks:
        if os.path.exists(path):
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc}")
            all_present = False

    return all_present


def test_no_old_utils():
    """Test that old utils.py doesn't exist in root."""
    print("\n" + "=" * 60)
    print("Testing Old Files Removed")
    print("=" * 60)

    if not os.path.exists('utils.py'):
        print("✓ Old utils.py removed from root")
        return True
    else:
        print("✗ Old utils.py still exists in root")
        return False


def count_code_reduction():
    """Estimate code reduction from refactoring."""
    print("\n" + "=" * 60)
    print("Code Reduction Analysis")
    print("=" * 60)

    # Count lines in core modules
    core_files = [
        'core/data_loader.py',
        'core/pdf_handler.py',
        'core/search.py',
        'core/embeddings.py',
        'core/citations.py',
        'core/ui_helpers.py',
        'core/prompts.py',
    ]

    total_core_lines = 0
    for file_path in core_files:
        with open(file_path, 'r') as f:
            lines = len([l for l in f.readlines() if l.strip() and not l.strip().startswith('#')])
            total_core_lines += lines

    print(f"Total non-comment lines in core/: {total_core_lines}")

    # Estimate old code (preserved in utils_old.py and legacy)
    if os.path.exists('utils_old.py'):
        with open('utils_old.py', 'r') as f:
            old_utils_lines = len([l for l in f.readlines() if l.strip() and not l.strip().startswith('#')])
        print(f"Old utils.py: {old_utils_lines} lines")

    legacy_total = 0
    legacy_files = [
        'legacy/streamlit/utils.py',
        'legacy/serp_api_qa/utils.py',
    ]
    for file_path in legacy_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = len([l for l in f.readlines() if l.strip() and not l.strip().startswith('#')])
                legacy_total += lines
                print(f"{file_path}: {lines} lines")

    print(f"\nEstimated duplicate code eliminated: ~{legacy_total + old_utils_lines - total_core_lines} lines")

    return True


def main():
    """Run all structure tests."""
    print("\n")
    print("*" * 60)
    print("S2QA REFACTORING STRUCTURE TEST")
    print("*" * 60)
    print()

    tests = [
        ("Core Structure", test_core_structure),
        ("Python Syntax", test_syntax),
        ("Import Updates", test_imports_updated),
        ("Legacy Preservation", test_legacy_preserved),
        ("Old Files Removed", test_no_old_utils),
        ("Code Reduction", count_code_reduction),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
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

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All structure tests passed!")
        print("\nNote: Runtime tests require dependencies (pandas, numpy, etc.)")
        print("These dependencies will be available when the app runs normally.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

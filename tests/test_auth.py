"""Tests for the auth module — API key generation and verification."""

from __future__ import annotations

import os

from memory_core.auth import generate_api_key, verify_api_key


class TestGenerateApiKey:
    def test_returns_32_char_hex(self):
        key = generate_api_key()
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_unique_each_call(self):
        keys = {generate_api_key() for _ in range(20)}
        assert len(keys) == 20


class TestVerifyApiKey:
    def test_no_expected_key_always_passes(self):
        assert verify_api_key(None, expected="") is True
        assert verify_api_key("anything", expected="") is True
        assert verify_api_key(None, expected=None) is True

    def test_correct_key_passes(self):
        assert verify_api_key("secret123", expected="secret123") is True

    def test_wrong_key_fails(self):
        assert verify_api_key("wrong", expected="secret123") is False

    def test_none_provided_with_expected_fails(self):
        assert verify_api_key(None, expected="secret123") is False

    def test_empty_provided_with_expected_fails(self):
        assert verify_api_key("", expected="secret123") is False

    def test_uses_env_var_as_default(self, monkeypatch):
        monkeypatch.setenv("CLICKMEM_API_KEY", "from-env")
        assert verify_api_key("from-env") is True
        assert verify_api_key("wrong") is False

    def test_timing_safe_comparison(self):
        key = generate_api_key()
        assert verify_api_key(key, expected=key) is True
        assert verify_api_key(key + "x", expected=key) is False

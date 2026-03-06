"""Tests for the mDNS discovery module."""

from __future__ import annotations

import socket

import pytest

from memory_core.discovery import get_local_ip, SERVICE_TYPE, SERVICE_NAME

try:
    from memory_core.discovery import register_service, discover, discover_one
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False


class TestGetLocalIp:
    def test_returns_string(self):
        ip = get_local_ip()
        assert isinstance(ip, str)

    def test_looks_like_ip(self):
        ip = get_local_ip()
        parts = ip.split(".")
        assert len(parts) == 4
        for p in parts:
            assert p.isdigit()
            assert 0 <= int(p) <= 255


class TestServiceConstants:
    def test_service_type(self):
        assert "_clickmem._tcp.local." in SERVICE_TYPE

    def test_service_name(self):
        assert "ClickMem" in SERVICE_NAME


@pytest.mark.skipif(not HAS_ZEROCONF, reason="zeroconf not installed")
class TestRegisterAndDiscover:
    def test_register_returns_cleanup(self):
        ip = get_local_ip()
        cleanup = register_service(ip, 19527)
        assert callable(cleanup)
        cleanup()

    def test_discover_returns_list(self):
        results = discover(timeout=0.5)
        assert isinstance(results, list)

    def test_discover_one_returns_none_or_str(self):
        result = discover_one(timeout=0.5)
        assert result is None or isinstance(result, str)

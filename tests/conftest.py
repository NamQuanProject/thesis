"""Tests are small, isolated, and offline by default."""
import socket
import pytest
import torch


@pytest.fixture(autouse=True)
def isolated_config_and_network(monkeypatch):
    from lib.config import reset_config
    reset_config()
    torch.set_num_threads(2)
    def offline(*args, **kwargs):
        raise RuntimeError('Tests must not download weights or datasets')
    monkeypatch.setattr(socket.socket, 'connect', offline)
    yield
    reset_config()

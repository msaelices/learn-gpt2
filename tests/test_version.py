"""Test version information."""

import gpt2


def test_version():
    """Test that version is defined."""
    assert hasattr(gpt2, "__version__")
    assert isinstance(gpt2.__version__, str)

import pytest
from unittest.mock import patch
import sys
from scripts.pindf_cli import main

def test_cli_help(capsys):
    # Test that the CLI can print help without crashing
    with patch.object(sys, 'argv', ['pindf_cli.py', '--help']):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
        
    out, err = capsys.readouterr()
    assert "PI-NDF High-Resolution Cube Exporter" in out
    assert "--atom1" in out

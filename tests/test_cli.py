from pyising.cli import main


def test_list(capsys):
    assert main(["list"]) == 0
    assert "numpy" in capsys.readouterr().out


def test_run_numpy(capsys):
    assert main(["run", "--backend", "numpy", "--L", "32", "--sweeps", "5", "--warmup", "1",
                 "--T", "2.0"]) == 0
    assert "flips/ns" in capsys.readouterr().out

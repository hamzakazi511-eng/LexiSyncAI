import logging
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.getLogger("lexisync").setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("samples")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_sample_docs.py"), str(out)], check=True,
                   capture_output=True)
    return out


@pytest.fixture(scope="session")
def case(sample_dir, tmp_path_factory):
    from lexisync.store import Case

    c = Case.create("test case", root=tmp_path_factory.mktemp("cases"))
    for f in sorted(sample_dir.glob("*.pdf")):
        c.add_pdf(f)
    c.build_index(train_autoencoder=True)
    return c

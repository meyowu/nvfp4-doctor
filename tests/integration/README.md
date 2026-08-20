# Integration Tests

GPU and framework integration tests live here and must declare their hardware
and software prerequisites.

E002's checked-in GPU differential is run explicitly rather than during the CPU
pytest suite:

```bash
source ./activate-nvfp4-lab.sh
PYTHONPATH=src python scripts/run_e002_gate1.py
```

It requires the pinned RTX 5080 / `sm_120` environment and rewrites only the
small E002 manifest and results files.

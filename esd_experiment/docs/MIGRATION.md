# Migration Guide: Old Structure → New Structure

This guide explains the reorganization and how to update your workflows.

## What Changed?

The repository was reorganized from a flat structure to a clean, segregated structure with dedicated directories for different components.

### Before (v1.0.1 and earlier)

```
esd_experiment/
├── run_esd_experiment.py
├── esd_worker.py
├── model_loader.py
├── analyze_results.py
├── create_model_list.py
├── test_setup.py
├── test_gpu.py
├── config_example.py
├── example_workflow.sh
├── sample_models.csv
├── README.md
├── QUICKSTART.md
├── OVERVIEW.md
├── GPU_FIX.md
├── requirements.txt
└── CHANGELOG.md
```

### After (v1.1.0+)

```
esd_experiment/
├── src/                      # Core framework
│   ├── run_experiment.py
│   ├── worker.py
│   └── model_loader.py
│
├── utils/                    # Utilities
│   ├── analyze_results.py
│   └── create_model_list.py
│
├── docs/                     # Documentation
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── OVERVIEW.md
│   └── GPU_FIX.md
│
├── examples/                 # Examples
│   ├── workflow.sh
│   ├── config.py
│   └── example_models.csv
│
├── tests/                    # Tests
│   ├── test_setup.py
│   └── test_gpu.py
│
├── run_experiment.py         # Wrapper
├── analyze_results.py        # Wrapper
├── create_model_list.py      # Wrapper
├── README.md
├── STRUCTURE.md
├── requirements.txt
└── CHANGELOG.md
```

## File Mapping

| Old Location | New Location | Notes |
|-------------|-------------|-------|
| `run_esd_experiment.py` | `src/run_experiment.py` | Renamed |
| `esd_worker.py` | `src/worker.py` | Renamed |
| `model_loader.py` | `src/model_loader.py` | Moved |
| `analyze_results.py` | `utils/analyze_results.py` | Moved |
| `create_model_list.py` | `utils/create_model_list.py` | Moved |
| `test_setup.py` | `tests/test_setup.py` | Moved |
| `test_gpu.py` | `tests/test_gpu.py` | Moved |
| `config_example.py` | `examples/config.py` | Renamed |
| `example_workflow.sh` | `examples/workflow.sh` | Renamed |
| `sample_models.csv` | `examples/example_models.csv` | Renamed |
| `results/model--name.csv` | `results/stats/model--name.csv` | Output path updated |
| `results/failed_models.txt` | `results/logs/failed_models.txt` | Failure log path updated |
| `README.md` | `docs/README.md` | Copied (+ new root README) |
| `QUICKSTART.md` | `docs/QUICKSTART.md` | Moved |
| `OVERVIEW.md` | `docs/OVERVIEW.md` | Moved |
| `GPU_FIX.md` | `docs/GPU_FIX.md` | Moved |

## Do You Need to Change Your Commands?

**NO!** Wrapper scripts maintain backward compatibility.

### Commands That Still Work

```bash
# These commands work exactly the same:
python run_experiment.py --model_list models.csv --output_dir results/ --gpus 0
python analyze_results.py --results_dir results/ --verbose
python create_model_list.py create --output models.csv
python tests/test_setup.py
python tests/test_gpu.py
```

The wrapper scripts at the root automatically call the correct implementation in subdirectories.

## Updating Custom Scripts

If you have custom scripts that import from the old locations, update them:

### Importing model_loader

**Old:**
```python
from model_loader import load_model, parse_model_string
```

**New:**
```python
from src.model_loader import load_model, parse_model_string
# OR
import sys
sys.path.insert(0, 'src')
from model_loader import load_model, parse_model_string
```

### Running Worker Directly

**Old:**
```bash
python esd_worker.py --model_id "meta-llama/Llama-2-7b-hf" --output_dir results/
```

**New:**
```bash
python src/worker.py --model_id "meta-llama/Llama-2-7b-hf" --output_dir results/
```

### Calling from Python

**Old:**
```python
import subprocess
subprocess.run(["python", "esd_worker.py", "--model_id", model])
```

**New:**
```python
import subprocess
subprocess.run(["python", "src/worker.py", "--model_id", model])
```

## Updating Workflows

### Example Workflow Script

**Old:**
```bash
python run_esd_experiment.py \
    --model_list models.csv \
    --output_dir results/ \
    --gpus 0 1 2 3
```

**New (still works!):**
```bash
python run_experiment.py \
    --model_list models.csv \
    --output_dir results/ \
    --gpus 0 1 2 3
```

### Config Files

**Old:**
```python
# config_example.py
cmd = "python run_esd_experiment.py ..."
```

**New:**
```python
# examples/config.py
cmd = "python run_experiment.py ..."
```

### Output Paths

Per-model and failure files moved to dedicated subdirectories:

- `results/stats/*.csv` for per-model result CSVs
- `results/logs/failed_models.txt` for failure tracking

Compatibility behavior:
- `src/run_experiment.py` checks both current and legacy locations when resuming
- `utils/analyze_results.py` reads from `results/stats/` first, then falls back to legacy root CSVs

## Documentation Updates

### Finding Documentation

**Old paths:**
```
README.md
QUICKSTART.md
OVERVIEW.md
GPU_FIX.md
```

**New paths:**
```
README.md               # Quick overview (NEW)
docs/README.md          # Full guide
docs/QUICKSTART.md      # Quick start
docs/OVERVIEW.md        # Architecture
docs/GPU_FIX.md         # GPU troubleshooting
STRUCTURE.md            # Structure explanation (NEW)
```

### Documentation Hierarchy

```
Start Here: README.md (root)
    ↓
Quick Start: docs/QUICKSTART.md
    ↓
Full Guide: docs/README.md
    ↓
Architecture: docs/OVERVIEW.md
```

## Benefits of New Structure

### 1. **Cleaner Root Directory**
- Only 7 files in root (down from 14+)
- Easy to see what's important

### 2. **Clear Organization**
- Core code in `src/`
- Utilities in `utils/`
- Documentation in `docs/`
- Examples in `examples/`
- Tests in `tests/`

### 3. **Better Scalability**
- Easy to add new components
- Clear where things belong
- Reduced clutter

### 4. **Professional Structure**
- Follows Python best practices
- Similar to popular frameworks
- Easier for new users

## Checklist for Migration

If you're updating from the old structure:

- [ ] Pull latest changes
- [ ] Test basic commands still work
- [ ] Update any custom imports in your code
- [ ] Update any direct file references
- [ ] Update documentation links
- [ ] Update any automation/CI scripts
- [ ] Run `python tests/test_setup.py` to verify

## Backward Compatibility

### What's Guaranteed

✅ Root-level command wrappers (`run_experiment.py`, etc.)  
✅ Command-line argument names  
✅ Output file schema/format (paths updated; schema unchanged)  
✅ Model list CSV format  
✅ Result CSV structure  

### What Changed

⚠️ Internal import paths (use `src.`, `utils.`)  
⚠️ Direct file paths (now in subdirectories)  
⚠️ Script names (`run_esd_experiment` → `run_experiment`)  
⚠️ Default output locations (`results/stats/` and `results/logs/`)  

## Getting Help

### If Commands Don't Work

1. **Run test script:**
   ```bash
   python tests/test_setup.py
   ```

2. **Check you're in the right directory:**
   ```bash
   pwd  # Should end with /esd_experiment
   ls   # Should see src/, utils/, docs/, etc.
   ```

3. **Verify Python can find modules:**
   ```bash
   python -c "from src.model_loader import load_model; print('OK')"
   ```

### If Imports Fail

Add the correct directory to Python path:
```python
import sys
from pathlib import Path

# For src/ modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# For utils/ modules  
sys.path.insert(0, str(Path(__file__).parent / "utils"))
```

### If Documentation is Unclear

1. Start with root `README.md`
2. Check `STRUCTURE.md` for organization
3. Read `docs/QUICKSTART.md` for examples
4. Refer to `docs/README.md` for details

## Questions?

See:
- [STRUCTURE.md](../STRUCTURE.md) - Directory organization
- [docs/README.md](README.md) - Full documentation
- [docs/QUICKSTART.md](QUICKSTART.md) - Common tasks
- [CHANGELOG.md](../CHANGELOG.md) - What changed and when

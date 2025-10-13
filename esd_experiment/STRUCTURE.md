# Repository Structure

Clean, organized directory structure for the ESD experiment framework.

## Overview

```
esd_experiment/
│
├── 📁 src/                   Core framework code
├── 📁 utils/                 Utility scripts
├── 📁 docs/                  Documentation
├── 📁 examples/              Example files and workflows
├── 📁 tests/                 Testing scripts
│
├── 🐍 run_experiment.py      Main entry point
├── 🐍 analyze_results.py     Analysis entry point
├── 🐍 create_model_list.py   Model list entry point
│
├── 📄 README.md              Quick start guide
├── 📄 CHANGELOG.md           Version history
├── 📄 requirements.txt       Python dependencies
└── 📄 .gitignore            Git ignore patterns
```

## Directory Details

### 📁 `src/` - Core Framework

The heart of the experiment framework. Contains the main execution logic.

```
src/
├── __init__.py              Python package marker
├── run_experiment.py        Orchestrates the entire experiment
│                            - Reads model lists
│                            - Manages GPU allocation via gputracker
│                            - Dispatches worker processes
│
├── worker.py                Per-model analysis worker
│                            - Loads single model
│                            - Runs ESD analysis
│                            - Saves results
│                            - Handles errors and retries
│
└── model_loader.py          Model loading utilities
                             - Detects adapter vs standard models
                             - Loads and merges PEFT adapters
                             - Handles HuggingFace authentication
```

**Key Features:**
- GPU-aware execution
- Robust error handling
- Automatic resume capability
- Clean separation of concerns

### 📁 `utils/` - Utility Scripts

Helper scripts for working with experiments before and after execution.

```
utils/
├── __init__.py              Python package marker
│
├── analyze_results.py       Post-experiment analysis
│                            - Aggregates per-model results
│                            - Computes summary statistics
│                            - Generates reports
│
└── create_model_list.py     Model list management
                             - Convert from text/CSV formats
                             - Merge multiple lists
                             - Create templates
```

**Usage:**
```bash
# Create model list
python create_model_list.py create --output models.csv --models "org/model1" "org/model2"

# Analyze results
python analyze_results.py --results_dir results/ --verbose
```

### 📁 `docs/` - Documentation

Comprehensive documentation covering all aspects of the framework.

```
docs/
├── README.md                Full user guide
│                            - Detailed feature descriptions
│                            - Advanced usage patterns
│                            - Integration examples
│
├── QUICKSTART.md            5-minute quick start
│                            - Basic usage
│                            - Common workflows
│                            - Quick troubleshooting
│
├── OVERVIEW.md              Technical architecture
│                            - System design
│                            - Component interactions
│                            - Extension points
│
└── GPU_FIX.md               GPU setup and troubleshooting
                             - How GPU allocation works
                             - Common issues and solutions
                             - Performance benchmarks
```

**Documentation Hierarchy:**
1. Start with `README.md` (root) for overview
2. Read `docs/QUICKSTART.md` for quick start
3. Refer to `docs/README.md` for details
4. Check `docs/GPU_FIX.md` for GPU issues

### 📁 `examples/` - Examples

Ready-to-use examples and templates.

```
examples/
├── workflow.sh              Complete workflow script
│                            - Creates model list
│                            - Runs experiment
│                            - Analyzes results
│                            - Shows best practices
│
├── config.py                Configuration patterns
│                            - Preset configurations
│                            - Command builders
│                            - Reusable settings
│
└── example_models.csv       Sample model list
                             - Format example
                             - Mix of standard and adapter models
```

**Try It:**
```bash
cd examples
bash workflow.sh
```

### 📁 `tests/` - Testing

Scripts to verify framework setup and functionality.

```
tests/
├── test_setup.py            Framework setup verification
│                            - Check dependencies
│                            - Verify file structure
│                            - Test imports
│
└── test_gpu.py              GPU setup testing
                             - CUDA availability
                             - Model loading on GPU
                             - GPU assignment simulation
```

**Run Tests:**
```bash
python tests/test_setup.py    # Verify installation
python tests/test_gpu.py      # Check GPU setup
```

## Entry Points

### 🐍 Root-Level Scripts

Convenient entry points in the repository root. These are thin wrappers that call the actual implementation in `src/` or `utils/`.

**run_experiment.py** → `src/run_experiment.py`
```bash
python run_experiment.py --model_list models.csv --output_dir results/ --gpus 0 1 2 3
```

**analyze_results.py** → `utils/analyze_results.py`
```bash
python analyze_results.py --results_dir results/ --verbose
```

**create_model_list.py** → `utils/create_model_list.py`
```bash
python create_model_list.py create --output models.csv
```

**Why wrappers?**
- Clean, simple commands from repo root
- No need to remember subdirectory paths
- Internal organization without affecting UX

## Configuration Files

### 📄 `requirements.txt`

Python package dependencies:
```
torch>=2.0.0
transformers>=4.30.0
peft>=0.5.0
accelerate>=0.20.0
...
```

### 📄 `.gitignore`

Excludes from version control:
- Python bytecode (`__pycache__/`)
- Virtual environments (`venv/`, `env/`)
- Results and logs
- Model caches
- IDE files

### 📄 `CHANGELOG.md`

Version history and changes:
- Bug fixes
- New features
- Breaking changes

## Design Principles

### 1. **Separation of Concerns**
- `src/` = Core execution logic
- `utils/` = Helper tools
- `docs/` = Documentation
- `examples/` = Reference implementations
- `tests/` = Verification

### 2. **Progressive Disclosure**
- Root README = Quick overview
- docs/QUICKSTART = 5-minute start
- docs/README = Full details
- docs/OVERVIEW = Architecture

### 3. **Usability First**
- Simple commands from root
- No complex paths to remember
- Clear error messages
- Helpful defaults

### 4. **Maintainability**
- Each file has single purpose
- Clear naming conventions
- Well-commented code
- Comprehensive documentation

## File Naming Conventions

### Python Files
- `snake_case.py` for scripts
- `__init__.py` for packages
- Descriptive names (e.g., `model_loader.py` not `ml.py`)

### Documentation
- `ALLCAPS.md` for top-level docs (README.md, CHANGELOG.md)
- `TitleCase.md` for specific guides (QUICKSTART.md)
- `lowercase.md` for internal docs

### Examples
- `workflow.sh` for runnable scripts
- `config.py` for configuration
- `example_*.csv` for sample data

## Common Operations

### Adding a New Feature

1. **Core logic** → Add to `src/`
2. **Helper function** → Add to `utils/`
3. **Documentation** → Update `docs/README.md`
4. **Example** → Add to `examples/`
5. **Test** → Add to `tests/`

### Creating a New Utility

1. Create in `utils/new_utility.py`
2. Add entry point wrapper in root
3. Document in `docs/README.md`
4. Add example to `examples/`

### Writing Documentation

1. Quick usage → `README.md` (root)
2. Tutorial → `docs/QUICKSTART.md`
3. Reference → `docs/README.md`
4. Architecture → `docs/OVERVIEW.md`

## Comparison: Before vs After

### Before (Flat Structure)
```
esd_experiment/
├── run_esd_experiment.py
├── esd_worker.py
├── model_loader.py
├── analyze_results.py
├── create_model_list.py
├── test_setup.py
├── test_gpu.py
├── README.md
├── QUICKSTART.md
├── OVERVIEW.md
├── GPU_FIX.md
├── config_example.py
├── example_workflow.sh
├── sample_models.csv
├── requirements.txt
└── CHANGELOG.md
```

**Issues:**
- 14+ files in root directory
- Hard to distinguish core vs utility vs docs
- No clear organization
- Difficult to navigate

### After (Organized Structure)
```
esd_experiment/
├── src/              # Core (3 files)
├── utils/            # Utilities (2 files)
├── docs/             # Documentation (4 files)
├── examples/         # Examples (3 files)
├── tests/            # Tests (2 files)
├── 3 entry points
└── 3 config files
```

**Benefits:**
- Clear separation of concerns
- Easy to find what you need
- Scalable organization
- Professional structure

## Summary

The reorganized structure provides:

✅ **Clear Organization** - Each directory has a specific purpose  
✅ **Easy Navigation** - Find files quickly  
✅ **Better Scalability** - Easy to add new components  
✅ **Professional Look** - Clean, maintainable codebase  
✅ **User-Friendly** - Simple commands from root  

All while maintaining backward compatibility through wrapper scripts! 🎉

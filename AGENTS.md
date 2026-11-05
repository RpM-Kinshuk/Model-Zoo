# Code clarity

- Name new modules and helpers by their purpose so readers do not need the implementation plan to understand them. Prefer `row_block_worker.py` and `extract_row_block_evidence` over phase names such as `t1_worker` or relationship words such as `inline`.
- Use explicit local names where several artifacts or roles coexist: `request_path`, `result_path`, `model_options`, `dispatch_thread`, and `outcome`.
- Keep scheduling and extraction responsibilities clear in names and docstrings. Explain non-obvious limits and invariants briefly; avoid dense expressions that hide them.
- During readability refactors, preserve existing CLI flags, artifact filenames, schema fields and tier labels unless their migration is explicitly requested. Explain established terms such as T1 at those boundaries.

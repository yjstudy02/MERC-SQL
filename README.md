# MERC-SQL core implementation

This release contains the core inference modules for **MERC-SQL: Heterogeneous Execution Consensus with a Fixed Number of SQL Candidates**, manuscript applsci-4690084. It is a portable extraction of the archived implementation, not a new algorithm. Generation and selection defaults follow the reported implementation. Generated predictions should be used to reproduce archived results; regeneration can vary with software, weights and device conditions.

## Files

* `generate.py`: local-model candidate generation, original decoding seeds and SQL extraction; original archived merge command.
* `prompts.py`: model-specific prompt construction and SQL extraction.
* `consensus.py`: bounded read-only execution, typed result normalization, duplicate-preserving grouping and deterministic voting.
* `evaluate.py`: adapter for the original BIRD execution evaluator, with read-only database connections.

Code comments and docstrings have been removed. Prompts, regular expressions and executable string values have been preserved. No databases, reference SQL annotations, pretrained weights, personal email addresses or server-specific absolute paths are distributed here.

## Installation

Python 3.10 or later. Selection uses only Python's standard library. Generation additionally requires PyTorch, Transformers, accelerate and bitsandbytes. The original reported environment used PyTorch 2.5.1+cu121 and Transformers 4.57.6; install CUDA-compatible builds for your device. The original BIRD evaluator has its own dependencies and must be obtained separately from its official distribution.

## Candidate generation

Input JSONL records contain `uid`, `db_id`, `question`, optional `evidence`, and a locally resolved `db_path`. No gold SQL is needed for generation. Download the two model checkpoints separately and comply with their licenses.

```bash
python generate.py worker --model omni --model-path /path/to/OmniSQL-7B --inputs generation_inputs.jsonl --out generated --gpu 0 --shard 0 --nshards 2
python generate.py worker --model omni --model-path /path/to/OmniSQL-7B --inputs generation_inputs.jsonl --out generated --gpu 1 --shard 1 --nshards 2
python generate.py worker --model qwen --model-path /path/to/Qwen2.5-Coder-7B-Instruct --inputs generation_inputs.jsonl --out generated --gpu 0 --shard 0 --nshards 1
python generate.py merge --inputs generation_inputs.jsonl --out generated
```

The merge command expects two OmniSQL shards and one Qwen shard, as in the archived script. It exports per-model records, not the selection manifest. Generation remains 5 OmniSQL candidates plus 1 Qwen candidate so both the 5O baseline and 4O+1Q method can be evaluated. MERC-SQL uses only positions `[0,1,2,3,5]`.

## Selection

Input JSONL records contain `uid`, `db_id`, a relative `database_file`, and six `sql_candidates` ordered `[o0,o1,o2,o3,o4,a]`. The separately exported candidate archive already has this structure. Map `database_file` to the layout of your official database files if needed.

```bash
python consensus.py --inputs candidates/bird_candidates.jsonl --db-root /path/to/dev_databases --output bird_predictions.jsonl --composition 4O1Q
python consensus.py --inputs candidates/bird_candidates.jsonl --db-root /path/to/dev_databases --output bird_5O_predictions.jsonl --composition 5O
```

The selector retains empty results, excludes failed executions, ignores row order, preserves duplicates and column order, and resolves ties by candidate order. Defaults are 12 seconds and 50,000 rows. The all-invalid fallback is position zero.

## Evaluation

Provide evaluation JSONL with `uid`, selected `sql`, `gold_sql` and `db_path`. Obtain gold annotations and database files directly from the dataset provider. Point `--evaluator-dir` to the directory containing the original `evaluation_ex.py` and `evaluation_utils.py`.

```bash
python evaluate.py --inputs evaluation_inputs.jsonl --evaluator-dir /path/to/bird/evaluation --output ex_results.jsonl
```

The adapter uses the original 30-second single-database-instance evaluator. Spider scores evaluated through this adapter are local EX scores, not official Spider test-suite submissions. BGE training, checkpoints, development-history logs and unrelated exploratory systems are outside this core-only release.

## Provenance and third-party resources

The prompt literal and generation helper functions were retained from the experiment archive; the OmniSQL prompt is the experiment's upstream-model prompt. Consult the OmniSQL upstream repository and its terms before redistribution. BIRD, Spider, OmniSQL, Qwen2.5-Coder, Transformers, PyTorch and bitsandbytes remain governed by their respective licenses. No upstream evaluator implementation or model logo is bundled. This package does not grant rights to third-party assets.

Suggested official resources:

* https://bird-bench.github.io/
* https://yale-lily.github.io/spider
* https://github.com/RUCKBReasoning/OmniSQL
* https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct

Choose a repository license for code you have the right to license before announcing an open-source release. This preparation package intentionally does not invent a third-party license grant. Before uploading to a public repository, review institutional/coauthor approval and the upstream prompt's redistribution terms.

## Archived candidate predictions

The candidates directory contains 1,534 BIRD and 2,147 Spider records, each ordered [o0,o1,o2,o3,o4,a], and revision evaluation summaries. Summary tables mix full-set and 200-question supplementary configurations: consult N, pool and protocol before comparing. Predictions do not include benchmark gold annotations, database contents or model weights. Use the official dataset distributions.

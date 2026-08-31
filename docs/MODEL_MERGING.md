# Model Merging

`authorship/tools/model_merge.py` wraps [`mergekit`](https://github.com/arcee-ai/mergekit)'s
`mergekit-pytorch` backend so that **raw training checkpoints** can be merged
directly. Merging several checkpoints (e.g. different training runs, or
S@8-best and EER-best checkpoints from the same run) often yields a model that
beats any single input — the `outputs/merged-*` checkpoints referenced
throughout these docs were produced this way.

> **Why a wrapper?** `mergekit` normally expects HuggingFace model
> directories. This project's checkpoints are raw PyTorch state dicts
> (`.ckpt` / `.pt`) containing optimizer and scheduler state as well as
> weights. The tool strips them to model-only state dicts in a `prepared/`
> subdirectory, writes a matching mergekit YAML, and invokes the backend.

## Usage

```bash
conda run -n hiatus-phase3 python -m authorship.tools.model_merge \
  --models '["/path/to/model_a.pt", "/path/to/model_b.pt"]' \
  --weights '[0.5, 0.5]' \
  --config-path configs/mergekit/model_merging.yaml \
  --output-dir outputs/merged-model \
  --run-merge
```

**Without `--run-merge` the command only prepares the config and prints the
merge command** — a safe dry run. Add the flag once the printed command looks
right.

| Arg | Default | Description |
|---|---|---|
| `--models` | *(required)* | JSON list of checkpoint paths |
| `--weights` | *(required)* | JSON list of merge weights, aligned with `--models` |
| `--output-dir` | *(required)* | Destination for the merged model |
| `--config-path` | *(required)* | Path the generated mergekit YAML is written to |
| `--base-model` | `None` | Base checkpoint, required by task-arithmetic methods (`ties`, `dare_ties`, …) |
| `--merge-method` | `linear` | Any `mergekit` merge method |
| `--density` | `None` | JSON list density gradient for sparsifying methods, e.g. `"[1,0.7,0.1]"` |
| `--dtype` | `float32` | Output dtype |
| `--no-normalize` | off | Disable weight normalization |
| `--prepared-dir` | `<output-dir>/prepared` | Where model-only checkpoints are staged |
| `--run-merge` | off | Actually execute the merge (default: print only) |
| `--no-cuda` | off | Don't pass `--cuda` to `mergekit-pytorch` |

### TIES merge with a base model

```bash
conda run -n hiatus-phase3 python -m authorship.tools.model_merge \
  --models '["/path/to/model_a.pt", "/path/to/model_b.pt"]' \
  --weights '[0.5, 0.5]' \
  --base-model "/path/to/base_model.pt" \
  --merge-method "ties" \
  --density '[1, 0.7, 0.1]' \
  --config-path "configs/mergekit/model_merging.yaml" \
  --output-dir "outputs/merged-model" \
  --run-merge
```

`scripts/merge_models.sh` holds this same example as an editable template —
adjust the paths and weights in place, or call the module directly.

## Generated config

`configs/mergekit/model_merging.template.yaml` shows the shape of the YAML
written to `--config-path`:

```yaml
models:
  - model: /path/to/model_a.safetensors
    parameters:
      weight: 0.5
  - model: /path/to/model_b.safetensors
    parameters:
      weight: 0.5
merge_method: linear
parameters:
  normalize: true
dtype: float32
```

The file at `--config-path` is **generated and overwritten** on each run; the
`.template.yaml` is a reference only and is not read by the tool.

## Using a merged checkpoint

Merged output is a directory of sharded safetensors. Point any consumer at
either the directory or its index file:

- Inference — `embedder_checkpoint_path="outputs/merged-model"` ([INFERENCE.md](INFERENCE.md#loading))
- Dense mining — `EMBEDDER_CONFIG_DIR=outputs/merged-model` ([PREPROCESSING.md](PREPROCESSING.md#2-dense-hard-pair-mining))
- Evaluation — `--embedder_checkpoint_path outputs/merged-model` ([EVALUATION.md](EVALUATION.md))

Copy the training `config.yaml` into the merged directory — downstream tools
resolve model architecture from `<dir>/config.yaml`, and the merge step does
not carry it over.

> Merging large checkpoints is memory- and storage-heavy (an 8B merge stages a
> full model-only copy per input under `prepared/`). Check free space before
> running, and delete `prepared/` once the merge succeeds.

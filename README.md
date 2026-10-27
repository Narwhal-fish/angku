# mlManager

Train models so that anyone can get the same result again.

mlManager runs your training experiments, locally or on a Kubernetes cluster with Ray, and
leaves a **bundle** for every run: the exact settings, the metrics of every epoch, the
environment it ran in, and the trained weights. Hand the bundle to someone else and they can
re-run it and check that they get the same run, the same metrics and bit-identical weights.

```
$ mlops reproduce a9253fa46953... --artifacts runs
tiny-cnn-smoke: REPRODUCED
  run_id         ok
  history        ok
  weights        ok
```

It was built for medical imaging research (MONAI, SwinUNETR, the MSD Spleen dataset), but works
with any PyTorch model.

## What it does for you

- **One file describes the experiment.** A YAML file names the code commit, the training image,
  the dataset version, the model, the loss, and every training setting including the seed.
  Vague versions such as `latest` or `main` are refused, because they would not mean the same
  thing next month.
- **Every experiment gets an id.** Two runs with the same id used the same inputs. Changing a
  description or a tag does not change the id; changing a learning rate does.
- **It checks instead of trusting.** Each run records the software and hardware it really ran
  on, and warns when that does not match what the YAML says. If the data on disk is not the
  data the YAML names, the run does not start.
- **Same inputs, same result.** Training is deterministic by default. A run that crashes and is
  resumed from its checkpoint ends exactly like one that was never interrupted.

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -e ".[tracking,tui]" --torch-backend=auto
```

Add only what you need:

| Option | Gives you |
|---|---|
| `tracking` | Log runs to MLflow |
| `tui` | `mlops ui`, the terminal browser for your runs |
| `distributed` | Multi-worker training with Ray, locally or on Kubernetes |
| `medical` | MONAI and NIfTI support for the medical imaging models and datasets |

For example `".[tracking,tui,distributed,medical]"` installs everything.

## Your first run

The repository ships a tiny experiment on synthetic data that finishes in seconds on a laptop:

```bash
mlops run experiments/configs/tiny_cnn_smoke.yaml --artifacts runs --tracker inmemory
```

```
tiny-cnn-smoke: FINISHED
  experiment_id  a78a4e510aa1fe85f284ceebbaa0fa85fe03b97fd0610727ffc4af86e9d20bc2
  run_id         a9253fa46953f7d0a28bef9350db5aacac74537ddee32917da0f67651c5c1156  (attempt 1, cpu, launcher local)
  epoch   1  train/loss=0.6405  val/loss=0.6355
```

You will also see a few `warning: not verified: ...` lines. The example points at a made-up
commit and image and uses generated data, so mlManager tells you what it could not confirm
instead of pretending. With your own experiment pinned to a real commit and real data, those
checks pass; the image check passes when the run happens inside the pinned image, as it does on
Kubernetes.

Now look at what you have:

```bash
mlops list --artifacts runs                         # your runs, grouped by experiment
mlops report <run_id> --artifacts runs              # a readable report of one run
mlops reproduce <run_id> --artifacts runs --tracker inmemory
mlops ui --artifacts runs                           # browse everything in the terminal
```

`<run_id>` can be copied from the output of `mlops run` or `mlops list`.

## Browse your runs

```bash
mlops ui --artifacts runs
```

A full-screen browser in your terminal: experiments, then their runs, then one run with tabs for
metrics, environment and the report. Press `?` for the keys. You can also:

- mark two runs with `space` and press `d` to see why they differ,
- press `r` to reproduce a finished run, or `R` to resume an interrupted one,
- watch runs that are still training update live,
- press `j` to see your jobs on a Kubernetes cluster and read their logs.

## Write your own experiment

Copy `experiments/configs/tiny_cnn_smoke.yaml` and edit it. Models, losses, augmentations,
metrics and datasets are referred to by name. To add your own model, put a file under
`experiments/models/`:

```python
from pydantic import Field
from torch import nn
from mlops.registry import MODELS, Params

class MyNetParams(Params):
    channels: int = Field(gt=0)

@MODELS.register("my_net", params=MyNetParams)
def build_my_net(p: MyNetParams) -> nn.Module:
    ...
```

and use it from the YAML:

```yaml
model:
  name: my_net
  params:
    channels: 16
```

A misspelled setting (`chanels: 16`) is an error, not silently ignored. Losses, augmentations,
metrics and datasets work the same way; `experiments/` has working examples of each, and
`experiments/configs/swinunetr_baseline.yaml` is a full medical imaging setup.

### Choose how the data is split

By default each dataset uses its own train/validation split. To choose it yourself, add a
`split` section:

```yaml
split:
  kind: holdout      # or kfold
  val: 0.2           # share of cases for validation
  test: 0.15         # optional: cases kept out of training and model selection entirely
```

or, for cross-validation, one experiment per fold:

```yaml
split:
  kind: kfold
  k: 5
  fold: 0            # 0 to 4; run each fold as its own experiment
```

Cases are shuffled with a fixed seed (`seed: 0`, separate from the training seed), so the same
YAML always gives the same split. Every run saves which case went where in `split.json`, and
`mlops diff` lists the cases that moved between two runs.

To pin your data, run `mlops snapshot <yaml> --data-root DIR` and put the printed hash in the
YAML's `dataset.snapshot`.

## What a run leaves behind

```
runs/<run_id>/
  spec.json        the experiment exactly as it ran
  manifest.json    ids, status, metrics per epoch, environment, data check
  model.pt         the trained weights
  best.pt          the weights of the best epoch (with --keep-best)
  split.json       which case was in train, validation and test (with a `split` section)
  checkpoint.pt    the latest checkpoint (with --checkpoint-every; removed once the run finishes)
  report.md        a readable summary of all of the above
```

This folder is everything someone else needs to reproduce the run.

## Evaluate a trained model

```bash
mlops predict <run_id> --artifacts runs
```

This applies the run's model (the best epoch's weights if you used `--keep-best`) to the test
cases of its `split` and reports the loss and metrics, per case and on average. The run itself is
never changed; the result goes into its own folder next to the runs, with a `report.md`. Running
the same command again gives exactly the same numbers.

Use `--part val` when the experiment has no test split, but keep in mind that validation data
was used during training, so those numbers look better than they really are. Add
`--save-predictions` to also keep each case's predicted labels. `mlops results --artifacts runs`
lists everything you have evaluated.

### Combine several models

```bash
mlops ensemble <run_a> <run_b> <run_c> --artifacts runs --vote soft
```

`--vote soft` averages the models' probabilities; `--vote hard` lets each model vote for a label
and takes the majority. The runs must be measured the same way (same data, preprocessing, loss,
metrics and test cases); the models themselves may differ. The output shows the combined result
next to each model on its own, so you can see whether combining helped. The k folds of a k-fold
experiment make a natural ensemble: they share the same test cases.

### How sure is the model?

```bash
mlops predict <run_id> --artifacts runs --mc-samples 20
```

For a model trained with dropout (for example `dropout: 0.2` in the `tiny_cnn` params), this
predicts every case 20 times with dropout switched on and averages the results. Besides the usual
metrics you get `uncertainty/entropy` and `uncertainty/variance` per case: the higher they are,
the less the samples agreed, so cases with high values are the ones worth a second look.
With `--save-predictions` a voxel-by-voxel uncertainty map is stored for each case as well. The
samples are seeded (`--mc-seed`, default 0), so the same command gives the same numbers again.

## Commands

| Command | What it does |
|---|---|
| `mlops run <yaml>` | Train one run |
| `mlops resume <run_id>` | Continue an interrupted run from its last checkpoint |
| `mlops reproduce <run_id>` | Re-run a finished run and check the result is the same |
| `mlops list` | Your runs, grouped by experiment, and whether repeated runs agree |
| `mlops report <run_id>` | Write the readable report of a run |
| `mlops diff <run_a> <run_b>` | Explain why two runs differ |
| `mlops predict <run_id>` | Evaluate a finished run's model on its test cases |
| `mlops ensemble <run_id>...` | Combine several runs' predictions by voting |
| `mlops results` | List your prediction results |
| `mlops ui` | Browse your runs in the terminal |
| `mlops id <yaml>` | Show an experiment's id without running it |
| `mlops snapshot <yaml>` | Show the hash of your data, to put in the YAML |
| `mlops rayjob <yaml>` | Run the experiment on a Kubernetes cluster |
| `mlops rayjob-status <name>` | Check a job on the cluster |
| `mlops rayjob-logs <name>` | Read a job's log (`-f` to follow) |

Useful options for `mlops run`:

- `--checkpoint-every N` saves a checkpoint every N epochs so the run can be resumed.
- `--keep-best val/dice/mean:max` keeps the weights of the best epoch.
- `--tracker mlflow --mlflow-uri sqlite:///mlflow.db` logs to MLflow.
- `--launcher ray` trains with several workers.
- `--strict-env` refuses to start when the environment does not match the YAML.

`mlops <command> --help` lists every option.

## Run on Kubernetes

With [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) installed on your
cluster:

1. Build and push the training image. The script prints the image digest:

   ```bash
   scripts/build-image.sh REGISTRY/mlops-train:TAG --push
   ```

2. Put that digest and your commit in the YAML. Then submit the job:

   ```bash
   mlops rayjob my_experiment.yaml --artifacts-pvc my-artifacts --submit
   ```

The job runs the same training as on your machine, and the bundle lands on the `my-artifacts`
volume. Add `--checkpoint-every N --ray-storage-pvc my-ray-storage` to make the job resumable;
if it is stopped, `--submit` prints the `mlops rayjob --resume` command that continues it.

### Try it on your laptop

With Docker, [kind](https://kind.sigs.k8s.io/) and kubectl you can run a small cluster locally
(CPU only):

```bash
scripts/kind-up.sh       # start a local cluster with KubeRay
scripts/kind-verify.sh   # train on it, stop a job halfway, resume it, check the result
mlops ui --artifacts .kind/artifacts --kubectl "kubectl --context kind-mlops"
scripts/kind-down.sh     # remove the cluster when you are done
```

## Tests

```bash
uv pip install -e ".[dev,medical]" --torch-backend=auto
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python -m pytest -q
```

GitHub Actions runs them on every push and pull request, once with every extra installed and
once with only the core.

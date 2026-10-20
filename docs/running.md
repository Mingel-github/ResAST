# Running ResAST

Run commands from the repository root. The examples use Windows PowerShell and an NVIDIA GPU. Install the package as described in the [README](../README.md), preferably in a separate Python 3.10 environment.

On a CPU-only computer, install the CPU version of PyTorch and replace `--device cuda` with `--device cpu` in the commands below. The CPU path uses full precision; the GPU training path uses mixed precision, so retraining can produce different numerical results. Checking bundled predictions with `resast-reference --root .` requires no GPU or retraining.

## Prepare data

Download the pinned source into a new directory and verify it:

```console
git clone -c core.autocrlf=false https://github.com/aster-droide/feline-age-prediction.git data/raw/feline-age-prediction
git -C data/raw/feline-age-prediction checkout --detach 3d02295bef1500d2b2500a124596f77010181391
resast-prepare --root . --upstream data/raw/feline-age-prediction
```

The loader uses [data/manifest.csv](../data/manifest.csv) and [splits/roles.csv](../splits/roles.csv): 792 calls grouped into 111 analysis IDs. If audio is stored separately, add `--audio-dir <AudioCropped-directory>` to preparation and extraction commands.

## Prepare AST features

```console
huggingface-cli download MIT/ast-finetuned-audioset-10-10-0.4593 config.json preprocessor_config.json model.safetensors --revision f826b80d28226b62986cc218e5cec390b1096902 --local-dir data/models/ast
resast-extract --kind ast --upstream data/raw/feline-age-prediction --weights data/models/ast --output data/cache --device cuda
resast-extract --kind acoustic --upstream data/raw/feline-age-prediction --output data/cache --device cpu
```

Feature extraction is needed once. Existing output files are preserved; use a fresh output directory when regenerating a cache. To reuse local data, copy `configs/paper.json` to `configs/local.json` and update the paths. Paths resolve relative to the configuration file.

## Train one model

For the reference training environment, set four CPU threads before starting Python:

```powershell
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
```

The following example trains the full model on repeat 0, fold 0, seed 17:

```console
resast-train --config configs/paper.json --model resast --stage select --repeat 0 --fold 0 --seed 17 --output outputs/reproduction --device cuda
resast-lock --output outputs/reproduction --models resast --repeats 0 --folds 0 --seeds 17 --lock outputs/pilot_epochs.json
resast-train --config configs/paper.json --model resast --stage refit --repeat 0 --fold 0 --seed 17 --selection-lock outputs/pilot_epochs.json --output outputs/reproduction --device cuda
```

`select` chooses the epoch count using validation data. `refit` starts fresh, trains on training plus validation data for that count, and evaluates the test data.

Outputs are saved under `outputs/reproduction/<stage>/<model>/repeat_<r>/fold_<f>/seed_<s>/`. They include the checkpoint, training history, prediction CSVs and `summary.json`.

Use `ast`, `multiscale_ast` or `ast_acoustic` instead of `resast` for the other AST variants, changing the model name in all three commands. Use a separate epoch-lock filename when checking a different subset. Add `--resume` to a training command to verify and reuse an already completed fit; an incomplete fit restarts from its seed.

For the complete supplied configuration, run selection for repeats 0–2, folds 0–3 and seeds 17, 43, 101. After all 36 selection fits for a model, lock them together:

```console
resast-lock --output outputs/reproduction --models resast --lock outputs/all_epochs.json
```

Then run `refit` for those same coordinates using `outputs/all_epochs.json`. Keep code, data and configuration unchanged between selection and refitting.

## Evaluate and ensemble

After the four test folds are complete for a repeat and seed, combine their `cat_predictions.csv` files:

```console
resast-evaluate --predictions outputs/reproduction/refit/resast/repeat_0/fold_0/seed_17/cat_predictions.csv outputs/reproduction/refit/resast/repeat_0/fold_1/seed_17/cat_predictions.csv outputs/reproduction/refit/resast/repeat_0/fold_2/seed_17/cat_predictions.csv outputs/reproduction/refit/resast/repeat_0/fold_3/seed_17/cat_predictions.csv --roles splits/roles.csv --model resast --repeat 0 --seed 17 --output outputs/oof/resast/r0/s17
```

Repeat for seeds 43 and 101, then optionally average their probabilities:

```console
resast-ensemble --seed17 outputs/oof/resast/r0/s17/oof_predictions.csv --seed43 outputs/oof/resast/r0/s43/oof_predictions.csv --seed101 outputs/oof/resast/r0/s101/oof_predictions.csv --roles splits/roles.csv --repeat 0 --output outputs/ensemble/resast/r0
```

Repeat with the corresponding paths for repeats 1 and 2. Metrics are saved on a 0–1 scale.

## Audio demo

After training, replace `data/raw/example.wav` with a local audio file:

```console
resast-predict --audio data/raw/example.wav --checkpoint outputs/reproduction/refit/resast/repeat_0/fold_0/seed_17/model.pt --backbone data/models/ast --device cuda
```

The output contains probabilities for kitten, adult and senior for that call.

## Comparison models

VGGish129, Perch1281 and YAMNet use the author feature tables downloaded with the source. Install their optional dependency:

```console
python -m pip install -c requirements/reference-constraints.txt -e ".[author-baselines]"
```

Use model names `vggish129`, `perch1281` or `yamnet` in the selection, lock and refit commands. These TensorFlow heads run on CPU in the reference Windows environment; use `--device cpu`.

For CNN14 and AVES extraction, install:

```console
python -m pip install -c requirements/reference-constraints.txt -e ".[feature-baselines]"
```

Place `Cnn14_mAP=0.431.pth` from the [PANNs release](https://zenodo.org/records/3987831) in `data/models/cnn14/`. Place `aves-base-bio.torchaudio.pt` and `aves-base-bio.torchaudio.model_config.json` from the [AVES model directory](https://storage.googleapis.com/esp-public-files/ported_aves/) in `data/models/aves/`. Expected hashes are recorded in [configs/pretrained.json](../configs/pretrained.json).

```console
huggingface-cli download Simon-Kotchou/ssast-base-patch-audioset-16-16 config.json preprocessor_config.json model.safetensors --revision f67df8e895e787bad0aec434563e8f0a1f61c794 --local-dir data/models/ssast
resast-extract --kind cnn14 --upstream data/raw/feline-age-prediction --weights data/models/cnn14 --output data/cache --device cuda
resast-extract --kind aves --upstream data/raw/feline-age-prediction --weights data/models/aves --output data/cache --device cuda
resast-extract --kind ssast --upstream data/raw/feline-age-prediction --weights data/models/ssast --output data/cache --device cuda
```

Train these heads using model names `cnn14`, `aves` or `ssast` in the same training commands.

## Checks

```console
python -m pytest -q
resast-reference --root .
```

The first command runs component tests. The second checks the bundled reference predictions and recomputes their metrics without training. Each command also supports `--help` for its arguments.

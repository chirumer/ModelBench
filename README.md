# ModelBench

ModelBench is a local evaluation utility for running the SSR-Net age and gender models against bundled benchmark images from `UTKFace` and `FairFace`.

The app lives in `modelbench/` and depends on the `SSR-Net/` git submodule for pretrained weights and upstream reference code.

## Repository layout

- `modelbench/`: FastAPI app, dataset manifests, dataset assets, tests, and ingestion script.
- `SSR-Net/`: upstream SSR-Net repository kept as a submodule.

## Setup

Clone with submodules, then create a Python 3.11 environment:

```bash
git clone --recurse-submodules https://github.com/chirumer/ModelBench.git
cd ModelBench
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r modelbench/requirements.txt
```

If the submodule has not been initialized yet:

```bash
git submodule update --init --recursive
```

## Run

```bash
uvicorn modelbench.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Dataset assets

This repository includes `100` local images for each bundled dataset:

- `UTKFace`
- `FairFace`

To refresh the local dataset bundle:

```bash
python -m modelbench.ingest_hf_datasets --count 100 --overwrite
```

## Supported presets

- `WIKI`: WIKI age + WIKI gender weights
- `IMDB`: IMDB age + IMDB gender weights
- `MORPH`: MORPH2 age + MORPH gender weights

The app resolves these weights from `./SSR-Net/pre-trained/...`.

## Test

```bash
pytest modelbench/tests
```

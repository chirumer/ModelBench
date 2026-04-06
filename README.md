# ModelBench

ModelBench is a local evaluation utility for running SSR-Net and DeepFace age and gender inference against bundled benchmark images from `UTKFace` and `FairFace`.

The repository now runs as three isolated local services:

- `modelbench/`: the user-facing FastAPI app and dataset orchestrator
- `services/ssrnet_service/`: a wrapper around the `SSR-Net/` submodule
- `services/deepface_service/`: a wrapper around the `deepface/` submodule

## Repository layout

- `modelbench/`: FastAPI app, dataset manifests, dataset assets, tests, and ingestion script.
- `services/`: local wrapper services plus runtime config and launcher.
- `SSR-Net/`: upstream SSR-Net repository kept as a submodule.
- `deepface/`: upstream DeepFace repository kept as a submodule.

## One-command startup

The primary local dev flow is:

```bash
git clone --recurse-submodules https://github.com/chirumer/ModelBench.git
cd ModelBench
python3 scripts/dev.py
```

The launcher:

- validates Python `3.11.11` from pyenv
- creates isolated venvs under `.venvs/`
- installs each service's dependencies
- starts:
  - `modelbench` on `8000`
  - `ssrnet_service` on `8101`
  - `deepface_service` on `8102`
- waits for health checks before marking the app ready

If the submodules have not been initialized yet first:

```bash
git submodule update --init --recursive
```

DeepFace may download its own attribute model weights on first use.

## Start / Stop / Restart

Use the service control script when you want the stack in the background:

```bash
python3 scripts/stack.py start
python3 scripts/stack.py status
python3 scripts/stack.py restart
python3 scripts/stack.py stop
```

Notes:

- `start` reuses the existing venvs by default.
- `start --install` or `restart --install` will reinstall dependencies first.
- Background logs are written to `.run/modelbench-stack.log`.
- The launcher pid is stored in `.run/modelbench-stack.pid`.

## Manual setup

Each runnable component has its own environment and uses Python `3.11.11`.

Create the environments:

```bash
~/.pyenv/versions/3.11.11/bin/python -m venv .venvs/modelbench
~/.pyenv/versions/3.11.11/bin/python -m venv .venvs/ssrnet
~/.pyenv/versions/3.11.11/bin/python -m venv .venvs/deepface
```

Install dependencies:

```bash
.venvs/modelbench/bin/pip install -r modelbench/requirements.txt
.venvs/ssrnet/bin/pip install -r services/ssrnet_service/requirements.txt
.venvs/deepface/bin/pip install -r services/deepface_service/requirements.txt
.venvs/deepface/bin/pip install --no-deps -e ./deepface
```

Start the services manually:

```bash
.venvs/ssrnet/bin/python -m uvicorn services.ssrnet_service.app:app --host 127.0.0.1 --port 8101
.venvs/deepface/bin/python -m uvicorn services.deepface_service.app:app --host 127.0.0.1 --port 8102
.venvs/modelbench/bin/python -m uvicorn modelbench.main:app --host 127.0.0.1 --port 8000
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
- `DeepFace`: DeepFace age and gender analysis with MTCNN detection

The SSR-Net presets resolve their weights from `./SSR-Net/pre-trained/...` inside the SSR-Net service.
The DeepFace preset runs inside the DeepFace service and calls `DeepFace.analyze(...)` with age and gender actions only.

## Test

```bash
.venvs/modelbench/bin/python -m pytest modelbench/tests services
```

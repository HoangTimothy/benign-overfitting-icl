# benign-overfitting-icl

An independent repository based on Garg et al. (2022), modernized before adding benign overfitting experiments. The original project description is retained below. MIT attribution and license are preserved.

## Setup (modern environment)

Use Python 3.12 and uv 0.11.31. From the repository root:

```sh
uv sync --locked
uv run --locked pytest -q
cd src
uv run --locked python train.py --config conf/smoke.yaml
```

The smoke config performs 3 CPU training updates, saves checkpoints, disables W&B, and skips the full evaluation suite. Original task configs retain their scientific settings. W&B defaults to disabled; set `wandb.mode: online` plus your entity/project to enable uploads. Full evaluation after training remains enabled by default; `eval_after_train: false` skips it.

For notebooks: `uv sync --locked --extra notebooks`, then `uv run jupyter lab` from `src`. The original notebook is unchanged; evaluate it with locally downloaded checkpoints.

Device selection defaults to CUDA when available, otherwise CPU. Set `device: cpu` or `device: cuda` explicitly in YAML. This checkout is validated on Windows CPU. CUDA and Linux execution require validation on those machines. The lockfile uses PyPI wheels; CUDA availability depends on the platform/wheel. Consult the [PyTorch installer](https://pytorch.org/get-started/locally/) for your server, and record any wheel/index change in a separate commit and lock update.

## Review every changed line

The tag `garg-2022-baseline` is the unmodified upstream commit `338337192195827c58356f9b627e5f999ea703d2`. The branch `main` preserves this baseline; modernization is isolated on `modernization/2026` until reviewed.

```sh
git diff garg-2022-baseline...modernization/2026 -- src
git log --reverse --oneline garg-2022-baseline..modernization/2026
python scripts/export_review.py
```

Open `review/generated/line-by-line.html` for a side-by-side diff with original/new line numbers. `all-changes.patch` contains every committed change, including the dependency lock. Generated files are ignored and can be regenerated at any commit. See [the Vietnamese review ledger](review/CHANGES_VI.md) for purpose, implications, and review checklist. No benign overfitting changes have been introduced.

## Original upstream README (historical setup)

The instructions below describe the 2022 environment. Use the setup above for this repository.

This repository contains the code and models for our paper:

**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>

![](setting.jpg)

```bibtex
    @InProceedings{garg2022what,
        title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
        author={Shivam Garg and Dimitris Tsipras and Percy Liang and Gregory Valiant},
        year={2022},
        booktitle={arXiv preprint}
    }
```

## Getting started
You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. Download [model checkpoints](https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip) and extract them in the current directory.

    ```
    wget https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip
    unzip models.zip
    ```

3. [Optional] If you plan to train, populate `conf/wandb.yaml` with you wandb info.

That's it! You can now explore our pre-trained models or train your own. The key entry points
are as follows (starting from `src`):
- The `eval.ipynb` notebook contains code to load our own pre-trained models, plot the pre-computed metrics, and evaluate them on new data.
- `train.py` takes as argument a configuration yaml from `conf` and trains the corresponding model. You can try `python train.py --config conf/toy.yaml` for a quick training run.

# Maintainers
* [Shivam Garg](https://cs.stanford.edu/~shivamg/)
* [Dimitris Tsipras](https://dtsipras.com/)

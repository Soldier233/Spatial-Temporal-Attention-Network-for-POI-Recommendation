# STAN: Spatial-Temporal-Attention-Network-for-Next-Location-Recommendation
[[Paper](https://arxiv.org/abs/2102.04095)]. [[Oral Youtube](https://www.youtube.com/watch?v=ajNzESvOvzs)] or [[Oral Bilibili](https://www.bilibili.com/video/BV1WL411E7Qm?from=search&seid=7472683569881802215)]. [[Implementation through LibCity](https://github.com/LibCity/Bigscity-LibCity)].

Thank you for your interest in our work! Thank you for reporting possible bugs and please make sure you are forking the latest repo to avoid eariler bugs. Before asking questions regarding the codes or the paper, I strongly recommend you to read the FAQ first. You can also use the [LibCity](https://github.com/LibCity/Bigscity-LibCity) version.

## Description
Because of the huge memory of the location matrix, the running speed of STAN is extremely low. You can refer to the implementation of masked attention [[here](https://github.com/yingtaoluo/PyHealth/blob/master/pyhealth/models/sequence/dipole.py)] if you wish to rewrite your own codes. 

Divide the dataset into different proportions of users to test the performance and then average. 

This repo has been updated to run with modern Python/PyTorch and to expose `recall@5` and `recall@10` directly in training logs. The current workflow is:

1. Prepare `.npy` files from raw data if needed.
2. Run `load.py` to build the padded tensors and spatio-temporal matrices.
3. Run `train.py` to train and report recall metrics.

The training log now looks like:

```text
epoch:1, time:84.83, valid_recall@5:0.2000, valid_recall@10:0.2000
epoch:1, time:84.83, test_recall@5:0.0000, test_recall@10:0.2000
best_epoch:1, valid_recall@5:0.2000, valid_recall@10:0.2000
best_epoch:1, test_recall@5:0.0000, test_recall@10:0.2000
```

## Environment
The original paper code was developed with Python 3.7.2, CUDA 10.1 and PyTorch 1.7.1. This repo now also works with a modern `Python 3.11` environment.

For macOS / Apple Silicon:

```bash
conda create -n pytorch python=3.11 pip -y
conda activate pytorch
pip install torch torchvision joblib tqdm
```

Check whether `MPS` is available:

```bash
python -c "import torch; print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available()); print(torch.mps.device_count())"
```

If `torch.backends.mps.is_available()` prints `True`, you can train with `--device mps`. Otherwise use `--device cpu` or `--device auto`.

## Data Preparation
The repo keeps the original sample files:

- `data/NYC.npy`
- `data/NYC_POI.npy`
- `data/dataset_TSMC2014_NYC.csv` can also be used directly if you have the Kaggle export

Each row of `NYC.npy` is `[user id, check-in location id, time in minutes]`.

### Download raw data

```bash
bash download_data.sh
```

This downloads:

- `poidata.zip`
- `loc-gowalla_totalCheckins.txt.gz`

into `data/raw/`.

### Build `.npy` from raw data

For Gowalla:

```bash
python prepare_raw.py --dataset Gowalla --raw-dir ./data/raw --output-dir ./data --top-pois 5000
```

Notes:

- By default the script prefers `data/raw/poidata/Gowalla`, which is much lighter than the full SNAP file for this implementation.
- `--top-pois 5000` is recommended because STAN builds a full `L x L` location distance matrix.

For NYC:

```bash
python prepare_raw.py --dataset NYC --raw-dir ./data/raw --output-dir ./data
```

The script will first look for `data/dataset_TSMC2014_NYC.csv` and then fall back to `data/raw/dataset_TSMC2014_NYC.txt`.
If you do not have the raw NYC file, you can directly use the repo-provided `data/NYC.npy` and `data/NYC_POI.npy`.

## Implemented Dimensions
The current codebase covers the five dimensions below with different levels of fidelity depending on the dataset.

### NYC

- `Spatial`: explicit, from POI latitude/longitude
- `Temporal`: explicit, from timestamp intervals and weekly time embedding
- `Semantic`: explicit, from `venueCategoryId` / `venueCategory` when `dataset_TSMC2014_NYC.csv` is available
- `Personal`: explicit, from each user's prefix-history visit distribution
- `Social`: proxy, from similar-user preference aggregation based on POI overlap

### Gowalla

- `Spatial`: explicit, from POI latitude/longitude
- `Temporal`: explicit, from timestamp intervals and weekly time embedding
- `Semantic`: proxy, from POI co-occurrence within user trajectories
- `Personal`: explicit, from each user's prefix-history visit distribution
- `Social`: proxy, from similar-user preference aggregation based on POI overlap

Notes:

- `NYC` has stronger semantic support because the Kaggle / TSMC file includes venue category fields.
- `Gowalla` in the current pipeline does not include explicit POI category labels, so semantic information is derived from trajectory co-occurrence instead.
- Neither dataset is currently using an explicit friend graph in this repo, so the social branch is implemented as a user-similarity proxy rather than a true social-network edge model.

### Build processed tensors

Run `load.py` after the `.npy` files are ready:

```bash
python load.py --dataset NYC
python load.py --dataset Gowalla
```

This writes:

- `data/NYC_data.pkl`
- `data/Gowalla_data.pkl`

## Training
Small smoke test on Mac with `MPS`:

```bash
python train.py --dataset NYC --part 10 --epochs 1 --device mps
python train.py --dataset Gowalla --part 10 --epochs 1 --device mps
```

Larger run:

```bash
python train.py --dataset NYC --part 100 --epochs 100 --device mps
python train.py --dataset Gowalla --part 100 --epochs 100 --device mps
```

Useful flags:

- `--device auto|cpu|mps|cuda`
- `--part N` to train on the first `N` users
- `--epochs N`
- `--embed-dim N`
- `--resume` to continue from a saved checkpoint
- `--plot-records` to load `records` from the best checkpoint and save recall curves
- `--plot-output PATH` to choose where the chart image is written

The trainer reports recall at `[1, 5, 10, 20]`, and prints `recall@5` and `recall@10` explicitly every epoch.

Plot curves directly from the saved best model:

```bash
python train.py --dataset NYC --device cpu --plot-records
python train.py --dataset NYC --device cpu --plot-records --plot-output ./nyc_records.png
```

This reads `records` from `best_stan_NYC.pth` (or the file given by `--checkpoint`) and writes a figure with validation and test recall curves.

## Benchmark
Evaluate a saved best checkpoint by running the model on the processed dataset and reporting `recall@5` / `recall@10`:

```bash
python benchmark.py --dataset NYC --device cpu
python benchmark.py --dataset Gowalla --device cpu --checkpoint ./best_stan_Gowalla.pth
```

This follows the repo's current split logic:

- validation uses each user's penultimate prediction step
- test uses each user's final prediction step

For a paper-style comparison with 10 independent runs and a paired T-test at `p=0.01`:

```bash
python benchmark.py \
  --dataset NYC \
  --device cpu \
  --checkpoint-glob 'runs/stan/*.pth' \
  --baseline-glob 'runs/baseline/*.pth'
```

Notes:

- `--checkpoint-glob` should match the 10 STAN checkpoints from 10 independent runs
- `--baseline-glob` should match the 10 checkpoints of the compared method in the same run order
- the paired T-test requires `scipy`

Run the whole paper-style workflow in one command:

```bash
python run_paper_benchmark.py --dataset NYC --device cpu --part 100 --epochs 10
```

This script will:

- train multiple seeds and save `best` checkpoints under `paper_runs/<dataset>/checkpoints/`
- benchmark each saved checkpoint with `recall@5` and `recall@10`
- compute mean/std over all runs
- save a per-run CSV and a summary JSON

To compare STAN against another method with a paired T-test:

```bash
python run_paper_benchmark.py \
  --dataset NYC \
  --device cpu \
  --part 100 \
  --epochs 10 \
  --baseline-glob 'runs/baseline/*.pth'
```

For a full paper-style reproduction run on a CUDA machine:

```bash
python run_paper_benchmark.py \
  --dataset NYC \
  --device cuda \
  --part -1 \
  --epochs 10 \
  --repeats 10 \
  --output-dir ./paper_runs/NYC_full_cuda
```

For Gowalla:

```bash
python run_paper_benchmark.py \
  --dataset Gowalla \
  --device cuda \
  --part -1 \
  --epochs 10 \
  --repeats 10 \
  --output-dir ./paper_runs/Gowalla_full_cuda
```

For a full paper-style comparison with paired T-test on a CUDA machine:

```bash
python run_paper_benchmark.py \
  --dataset NYC \
  --device cuda \
  --part -1 \
  --epochs 10 \
  --repeats 10 \
  --output-dir ./paper_runs/NYC_full_cuda \
  --baseline-glob 'runs/baseline/*.pth'
```

Notes:

- `--part -1` evaluates all processed users
- `--repeats 10` matches the paper's 10-run averaging protocol
- `summary.json` stores the averaged metrics, and `stan_runs.csv` stores each run separately
- use `--seeds 0,1,2,3,4,5,6,7,8,9` if you want to pin the exact seed list explicitly

## FAQs
Q1: Can you provide a dataset?  
A1: Our datasets are collected from the following links. Please feel free to do your own data processing on your model while comparing STAN as baseline.
http://snap.stanford.edu/data/loc-gowalla.html;  
https://personal.ntu.edu.sg/gaocong/data/poidata.zip;
http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip

The last link is currently unreliable in many environments. If you already have `data/NYC.npy` and `data/NYC_POI.npy`, you can skip raw NYC conversion and use them directly.  
  
Q2.1: What does it mean "The number of the training set is 𝑚 − 3, with the first 𝑚′ ∈ [1,𝑚 − 3] check-ins as input sequence and the [2,𝑚 − 2]-nd visited location as the label"?  
A2.1: We use [1] as input to predict [2], use [1,2] as input to predict [3], and ..., until we use [1,...,m-3] to predict [m-2]. Basically we do not use the last few steps and reserve them as a simulation of "future visits" to test the model since these last steps are not fed into the model during training.  
  
Q2.2: Can you please explain your trajectory encoding process? Do you create the location embeddings using skip-gram-like approaches?  
A2.2: Pre-training of embedding is an effective approach and can further improve the performance for sure. Unfortunately, the focus and contribution of this paper are not on embedding pre-training but on spatio-temporal linear embedding, and pretraining is not used in baselines, so we do not use it in our paper.

Q2.3: Would it be better to construct edges based on spatial distances instead of using distances?  
A2.3: If the edges can truly reflect the relations between each loaction and each user, then yes. Ideal 0-1 edge relation is a stronger representation. However, constructing edges merely based on spatial distances can raise problems. Consider that a 30-kilometer metro takes less time than a 5-kilometer walk. From the data, we only know distances.  

Q2.4: What do you mean by setting a unit spatiotemporal embedding?  
A2.4: ![image](https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation/blob/master/unit_embedding.png)

Q2.5: What does each column/row in NYC.npy mean?  
A2.5: Each row: [user id, check-in location id, time in minutes].  

Q2.6: Can we try a different division of train/dev/test datasets?  
A2.6: Our goal here is to generalize for the future visits of each user we have known (we do not want to test the model performance on biased past behavior), instead of generalizing to other users whose user-id embeddings are not known to the model. 

Q2.7: How is the value of the recall rate calculated in your paper? For example, the top5 probability of the NYC data set is 0.xx but in the paper it is 0.xxxx.  
A2.7: It is common practice to run under different seeds and get the average value. We averaged the ten times results and all of them are accepted by the statistical test of p=0.01. 

Q3: What is the environment to run the code? And version?  
A3: The original code was run with python 3.7.2, CUDA 10.1 and PyTorch 1.7.1. The current repo has also been tested with Python 3.11 and recent PyTorch builds on macOS. Make sure to install all imported libraries before running.  

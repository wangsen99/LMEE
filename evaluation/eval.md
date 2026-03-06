<h2 align="center">
  <b>Evaluation on LMEE-Bench</b>
</h2>

## Installation
Set up the conda environment (Linux, Python 3.9):
```bash
conda create -n lmee python=3.9 -y && conda activate lmee

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu121_pyt241.tar.bz2 -y

pip install -r requirements.txt

conda install -c conda-forge -c aihabitat habitat-sim=0.3.1 headless faiss-cpu=1.7.4 -y                                               
```

## Reasoning

### Step 1: Download Required Files
- Download **LMEE-Bench**: [LMEE-Bench Dataset](https://huggingface.co/datasets/wangsen99/LMEE-Bench)
- Download **MemoryExplorer Model**: [MemoryExplorer](https://huggingface.co/wangsen99/MemoryExplorer)

### Step 2: Configure Paths
Specify the paths in the configuration file: `cfg/eval_lmee_bench.yaml`.

### Step 3: Run the Reasoning Script
Execute the following command:

```bash
python run_lmee.py -cf cfg/eval_lmee_bench.yaml --answer_type open
```

- **answer_type**: Choose between `open` and `choice`.
- **Subset Options**:
  - `LMEE-Bench/lmee_bench_sub`: Includes 58 tasks.
  - `LMEE-Bench/lmee_bench`: Includes the full 166 tasks.

---

## Evaluation

### Step 1: Generate Results
After running the reasoning script, you will get the results file: `lmee_answer.json`.

### Step 2: Run the Evaluation Script
Use the following command to evaluate the question-answering performance:

```bash
python eval_lmee_bench.py --json_path "results/exp_eval_lmee/lmee_answer.json" --root_dir "../data/LMEE-Bench/task_test"
```

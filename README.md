<br/>
<p align="center">
  <h1 align="center">
    Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration
  </h1>
  <p align="center">
    <b>CVPR 2026</b>
  </p>
  <p align="center">
    Sen Wang,
    Bangwei Liu,
    Zhenkun Gao,
    Lizhuang Ma,
    Xuhong Wang,
    Yuan Xie,
  <a href="https://tanxincs.github.io">Xin Tan</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2601.10744">
    <img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=flat-square&logo=arxiv&logoColor=red">
    </a>
    <a href="https://wangsen99.github.io/papers/lmee/">
    <img src="https://img.shields.io/badge/Project-Page-4285F4?style=flat-square&logo=google-chrome&logoColor=blue">
    </a>
    <a href="https://huggingface.co/wangsen99/MemoryExplorer">
    <img src="https://img.shields.io/badge/Model-MemoryExplorer-orange?style=flat-square&logo=huggingface&logoColor=FFD21E">
    </a>
    <a href="https://huggingface.co/datasets/wangsen99/LMEE">
    <img src="https://img.shields.io/badge/Dataset-LMEE-green?style=flat-square&logo=huggingface&logoColor=FFD21E">
    </a>
    <a href="https://huggingface.co/datasets/wangsen99/LMEE-Bench">
    <img src="https://img.shields.io/badge/Benchmark-LMEE--Bench-8A2BE2?style=flat-square&logo=huggingface&logoColor=FFD21E">
    </a>
  </p>
</p>

---

This is the official repository of **Explore with Long-term Memory**: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration.

![](assets/intro.png)

---

## News
- [2026/03] Training code for MemoryExplorer is released.
- [2026/03] Inference code for LMEE-Bench is released.
- [2026/02] Our paper is accepted to CVPR 2026!
- [2026/01] [Paper](https://arxiv.org/abs/2601.10744) is on arXiv.

## Preparations
- (All) Download the train and val split of [HM3D-Sem](https://aihabitat.org/datasets/hm3d-semantics/)
- (Evaluation) Download **LMEE-Bench**: [LMEE-Bench Dataset](https://huggingface.co/datasets/wangsen99/LMEE-Bench) 
- (Evaluation) Download **MemoryExplorer Model**: [MemoryExplorer](https://huggingface.co/wangsen99/MemoryExplorer)
- (Training) Download **LMEE**: [LMEE Training Dataset](https://huggingface.co/datasets/wangsen99/LMEE)

Put them into the `data` folder. The final file format should be:
```shell
data
├── LMEE-train
│   ├── task_train
│   │   ├── easy
│   │   ├── hard
│   │   ├── medium
│   ├── train_data.parquet
│   │   
├── LMEE-Bench
│   ├── lmee_bench
│   ├── lmee_bench_sub
│   ├── task_test
│   │   ├── easy
│   │   ├── hard
│   │   ├── medium
│   │   
├── ...
```

## Evaluation
### Installation
Set up the conda environment (Linux, Python 3.9):
```bash
cd evaluation

conda create -n lmee python=3.9 -y && conda activate lmee
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu121_pyt241.tar.bz2 -y
pip install -r requirements.txt
conda install -c conda-forge -c aihabitat habitat-sim=0.3.1 headless faiss-cpu=1.7.4 -y
```

### Reasoning
Specify the paths in the configuration file: `cfg/eval_lmee_bench.yaml` and execute the following command:

```bash
python run_lmee.py -cf cfg/eval_lmee_bench.yaml --answer_type open
```
- **answer_type**: Choose between `open` and `choice`.
- **Subset Options**:
  - `LMEE-Bench/lmee_bench_sub`: Includes 58 tasks.
  - `LMEE-Bench/lmee_bench`: Includes the full 166 tasks.

### Evaluation
After running the reasoning script, you will get the results file: `lmee_answer.json` and use the following command to evaluate the question-answering performance:

```bash
python eval_lmee_bench.py --json_path "results/exp_eval_lmee/lmee_answer.json" --root_dir "../data/LMEE-Bench/task_test"
```

## Training
### Installation
Set up the conda environment (Linux, Python 3.10):
```bash
cd train

conda create -n memoryexplorer python=3.10 -y && conda activate memoryexplorer
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install hf_transfer vllm==0.8.5.post1 triton==3.2.0
pip install -e .
```

### Running
Specify the paths in the running file `train_my.sh` and `IMAGE_ROOT` in `verl\tooluse\memory_tool.py`, and execute the following command:
```bash
bash train_my.sh
```
Please see [EasyR1](https://github.com/hiyouga/EasyR1) and [verl](https://github.com/verl-project/verl) for more training details. 

## Todo List
- ~~Release training scripts and dataset~~
- Release data generation scripts

## Acknowledgement
The codebase is built upon [3D-Mem](https://github.com/UMass-Embodied-AGI/3D-Mem) and [MemoryEQA](https://github.com/memory-eqa/MemoryEQA).
We thank the authors for their great work.

## Citation
```tex
@inproceedings{wang2026explore,
  title={Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration},
  author={Wang, Sen and Liu, Bangwei and Gao, Zhenkun and Ma, Lizhuang and Wang, Xuhong and Xie, Yuan and Tan, Xin},
  booktitle={Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

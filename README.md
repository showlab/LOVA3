<p align="center">

  <h1 align="center">LOVA3: Learning to Visual Question Answering, Asking and Assessment</h1>
  <p align="center">
    <br>
        <a href="https://arxiv.org/abs/2405.14974"><img src='https://img.shields.io/badge/arXiv-LOVA3-red' alt='Paper PDF'></a>
        <a href='https://zhaohengyuan1.github.io/lova3.github.io/'><img src='https://img.shields.io/badge/Project_Page-LOVA3-green' alt='Project Page'></a>
        <a href="https://huggingface.co/hhenryz/LOVA3-llava-v1.5-7b"><img src='https://img.shields.io/badge/Model-LOVA3-blue' alt='Models'></a>
        <a href="https://huggingface.co/datasets/hhenryz/EvalQABench"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EvalQABench-yellow' alt='EvalQABench'></a>
        <a href="https://huggingface.co/datasets/hhenryz/Mixed_VQA_GenQA_EvalQA_1.5M"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TrainingData-yellow' alt='Dataset'></a>
    <br>
    <b>TL;DR: No hyperparameter modification and extra data annotation; LOVA3 is a new training paradigm for advancing multimodal training by incorporating new capabilities: asking questions and assessing vqa triplets.</b>
  </p>

</p>

### Overall Performance Improvements
<p align="center">
  <img src="./assets/comprehensive_comparison.png">
</p>

## Abstract

Question answering, asking, and assessment are three innate human traits crucial for understanding the world and acquiring knowledge. By enhancing these capabilities, humans can more effectively utilize data, leading to better comprehension and learning outcomes. However, current Multimodal Large Language Models (MLLMs) primarily focus on question answering, often neglecting the full potential of questioning and assessment skills. In this study, we introduce LOVA3 designed to equip MLLMs with these additional capabilities.

## 📢 Update
* [03/03/2025] We update four models in paper for testing, have fun!
* [10/16/2024] We release the [webpage](https://zhaohengyuan1.github.io/lova3.github.io/).
* [09/26/2024] LOVA3 is accepted by NeurIPS 2024.
* [07/01/2024] Related work [Genixer](https://github.com/zhaohengyuan1/Genixer) is accepted by ECCV 2024.
* [05/24/2024] We release the code of LOVA3, the [EvalQABench](https://huggingface.co/datasets/hhenryz/EvalQABench), the training dataset [Mixed_VQA_GenQA_EvalQA_1.5M.jsonl](https://huggingface.co/datasets/hhenryz/Mixed_VQA_GenQA_EvalQA_1.5M), and the checkpoint [LOVA3-llava-v1.5-7b](https://huggingface.co/hhenryz/LOVA3-llava-v1.5-7b).
* [05/23/2024] We release the LOVA3 [paper](https://arxiv.org/abs/2405.14974).

## 🌺 To Do List

- [x] Using Gemini-1.5-Flash to creating EvalQA training data with larger size and higher quality.

- [x] Applying LOVA3 to samller language model Phi-1.5.


<!-- ## 💡Key Contributions:

* **LOVA3** - To the best of our knowledge, LOVA3 is the first effort to imbue the asking and assessment abilities in training a robust and intelligent MLLM, inspired from human learning mechanism.
* **EvalQABench** - We build a new benchmark EvalQABench for the VQA correction evaluation as the first effort to advance the development of future research.

* **Performance Improvement** - Training with our proposed LOVA3 framework, we observe consistent improvement on 10 representative benchmarks.


**Usage and License Notices**: The data, and code is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. 

## GenQA: Learn to generate diverse VQA pairs for unlabeled images

If one MLLM is able to successfully generate high-quality question-answer pairs based on visual input, it indicates a stronger problem-solving ability. To enable the MLLM to ask questions, we carefully define five main multimodal data types as listed in following table.
<p align="center"><img src="./assets/GenQAData.png" alt="pipeline"/></p>


## EvalQA: Learn to assess the correctness of VQA triplet

### Automatic Data Generation Pipeline
Illustration of the proposed pipeline for generating negative answers and feedback.
<p align="center"><img src="assets/EvalqaPipeline.png" alt="pipeline"/></p>

### Selected examples from EvalQABench

<p align="center"><img src="assets/evalqa_visual.png" alt="pipeline"/></p>

### EvalQABench Results

<p align="center"><img src="assets/evalqabenchresult.png" alt="pipeline"/></p>

## Main Results

<p align="center"><img src="assets/result1.png" alt="pipeline"/></p>

<p align="center"><img src="assets/result2.png" alt="pipeline"/></p>

<p align="center"><img src="assets/result3.png" alt="pipeline"/></p> -->

## 🚀 Quick Start (Training)

If you are using the codebase [LLaVA](https://github.com/haotian-liu/LLaVA), just replace the `--data_path` with [Mixed_VQA_GenQA_EvalQA_1.5M.jsonl](https://huggingface.co/datasets/hhenryz/Mixed_VQA_GenQA_EvalQA_1.5M) to enjoy the performance improvement.

```bash
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./data/Mixed_VQA_GenQA_EvalQA_1.5M.jsonl \
    ...
```

## ⚒️ Install (Optional)

If you have the python environments for [LLaVA](https://github.com/haotian-liu/LLaVA), please skip this step.

```shell
conda create -n LOVA python=3.10
conda activate LOVA
pip install --upgrade pip
pip install -e .
```
## Model weights

|Model Name|Size|Checkpoint|EvalQA Data generated By|
|-|-|-|-|
|LOVA3-llava-v1.5-7b|7B|[checkpoint](https://huggingface.co/hhenryz/LOVA3-llava-v1.5-7b) | Fuyu-8B |
|LOVA3-llava-v1.5-7b-gemini|7B|[checkpoint](https://huggingface.co/ZechenBai/LOVA3-llava-v1.5-7b-gemini)| Gemini-1.5-Flash |
|LOVA3-llava-v1.5-phi1.5-baseline|1.5B|[checkpoint](https://huggingface.co/ZechenBai/LOVA3-llava-v1.5-phi1.5-baseline)| - |
|LOVA3-llava-v1.5-phi1.5-fuyu|1.5B|[checkpoint](https://huggingface.co/ZechenBai/LOVA3-llava-v1.5-phi1.5-fuyu) | Fuyu-8B |
|LOVA3-llava-v1.5-phi1.5-gemini|1.5B|[checkpoint](https://huggingface.co/ZechenBai/LOVA3-llava-v1.5-phi1.5-gemini)| Gemini-1.5-Flash |

Download from huggingface:
```
git clone https://huggingface.co/hhenryz/LOVA3-llava-v1.5-7b
```

## Data Preparation

### Download the data Json
* Training Data: [Mixed_VQA_GenQA_EvalQA_1.5M.jsonl](https://huggingface.co/datasets/hhenryz/Mixed_VQA_GenQA_EvalQA_1.5M).

* EvalQABench Data: [EvalQABench](https://huggingface.co/datasets/hhenryz/EvalQABench)

### Image Datasets

Please download the images from constituting datasets:

- COCO: [train2014](http://images.cocodataset.org/zips/train2014.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- AOKVQA: [download script](https://github.com/allenai/aokvqa?tab=readme-ov-file#downloading-the-dataset)
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- LLaVA-Instruct: [huggingface](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)


## 💃 Evaluation

1. Download [LOVA3-llava-v1.5-7b](https://huggingface.co/hhenryz/LOVA3-llava-v1.5-7b) under the folder `checkpoints`.

2. Download the CLIP vision encoder [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) under the folder `checkpoints`.

3. Run the evaluation scripts under the folder `scripts/v1_5/eval`. There are 12 multimodal datasets and benchmarks awaiting evaluation.

Take VizWiz as an example, the running command is as follows:

```
modelname=LOVA3-llava-v1.5-7b

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/$modelname \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /yourpath/vizwiz/test/ \
    --answers-file ./playground/data/eval/vizwiz/answers/$modelname.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$modelname.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$modelname.json

```

## Training

1. Download the pretrained MLP adapter weights [llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5) from and put it under the folder `checkpoints`.

2. Download the model weight [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) under the folder `checkpoints`.

3. Download the model weight [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) under the folder `checkpoints`.

4. Download the training data [Mixed_VQA_GenQA_EvalQA_1.5M.jsonl](https://huggingface.co/datasets/hhenryz/Mixed_VQA_GenQA_EvalQA_1.5M) under the folder `data`.

5. Run the training script.

```
bash scripts/v1_5/finetune.sh
```

## 🙏 Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): The codebase we built upon. 
- [LAVIS](https://github.com/salesforce/LAVIS): We download some datasets from its scripts.

## 🎓 Citation

If you find LOVA3 useful, please cite using this BibTeX:

```bibtex
@misc{zhao2024lova3learningvisualquestion,
      title={LOVA3: Learning to Visual Question Answering, Asking and Assessment}, 
      author={Henry Hengyuan Zhao and Pan Zhou and Difei Gao and Zechen Bai and Mike Zheng Shou},
      year={2024},
      eprint={2405.14974},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2405.14974}, 
}
```
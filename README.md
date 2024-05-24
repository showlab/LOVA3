# LOVA3: Learning to Visual Question Answering, Asking and Assessment

[Henry Hengyuan Zhao](https://github.com/zhaohengyuan1),  [Pan Zhou](https://panzhous.github.io), [Difei Gao](https://scholar.google.com/citations?user=No9OsocAAAAJ&hl=en) and [Mike Zheng Shou](https://sites.google.com/view/showlab)

## Abstract

Question answering, asking, and assessment are three innate human traits crucial for understanding the world and acquiring knowledge. By enhancing these capabilities, humans can more effectively utilize data, leading to better comprehension and learning outcomes. However, current Multimodal Large Language Models (MLLMs) primarily focus on question answering, often neglecting the full potential of questioning and assessment skills. In this study, we introduce LOVA3, an innovative framework named ``Learning tO Visual Question Answering, Asking and Assessment,'' designed to equip MLLMs with these additional capabilities. Our approach involves the creation of two supplementary training tasks GenQA and EvalQA, aiming at fostering the skills of asking and assessing questions in the context of images. To develop the questioning ability, we compile a comprehensive set of multimodal foundational tasks. For assessment, we introduce a new benchmark called EvalQABench, comprising 64,000 training samples (split evenly between positive and negative samples) and 5,000 testing samples. We posit that enhancing MLLMs with the capabilities to answer, ask, and assess questions will improve their multimodal comprehension and lead to better performance. We validate our hypothesis by training an MLLM using the LOVA3 framework and testing it on 10 multimodal benchmarks. The results demonstrate consistent performance improvements, thereby confirming the efficacy of our approach.

## 💡Key Contributions:

* **LOVA3** - To the best of our knowledge, LOVA3 is the first effort to imbue the asking and assessment abilities in training a robust and intelligent MLLM.
* **EvalQABench** - We build a new benchmark EvalQABench for the VQA evaluation as the first effort to advance the development of future research.


**Usage and License Notices**: The data, and code is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes. 

## GenQA
If one MLLM is able to successfully generate high-quality question-answer pairs based on visual input, it indicates a stronger problem-solving ability. To enable the MLLM to ask questions, we carefully define five main multimodal data types as listed in following table.
<p align="center"><img src="./assets/GenQAData.png" alt="pipeline"/></p>


## EvalQA

### Automatic Data Generation Pipeline
Illustration of the proposed pipeline for generating negative answers and feedback.
<p align="center"><img src="assets/EvalqaPipeline.png" alt="pipeline"/></p>

### Selected examples from EvalQABench

<p align="center"><img src="assets/samplesofevalqabench.png" alt="pipeline"/></p>

### EvalQABench Results

<p align="center"><img src="assets/evalqabenchresult.png" alt="pipeline"/></p>

## Main Results

<p align="center"><img src="assets/result1.png" alt="pipeline"/></p>

<p align="center"><img src="assets/result2.png" alt="pipeline"/></p>

<p align="center"><img src="assets/result3.png" alt="pipeline"/></p>

## Install

```shell
conda create -n LOVA python=3.10
conda activate LOVA
pip install --upgrade pip
pip install -e .
```
## Model weight

Pretrained weight: [LOVA3-llava-v1.5-7b](https://huggingface.co/hhenryz/LOVA3-llava-v1.5-7b)

Download it by using following command:

```
git clone https://huggingface.co/hhenryz/LOVA3-llava-v1.5-7b
```

## Training Data

* Here we provide the training/Evaluation/Testing sets of EvalQABench under the folder `EvalQABench`.

* Training data: [Mixed_VQA_GenQA_EvalQA_1.5M.jsonl](https://huggingface.co/datasets/hhenryz/Mixed_VQA_GenQA_EvalQA_1.5M).

### Image Datasets

Please download the images from constituting datasets:

- COCO: [train2014](http://images.cocodataset.org/zips/train2014.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- AOKVQA: [download script](https://github.com/allenai/aokvqa?tab=readme-ov-file#downloading-the-dataset)
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)
- LLaVA-Instruct: [huggingface](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)


## Evaluation

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

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): The codebase we built upon. 
- [LAVIS](https://github.com/salesforce/LAVIS): We download some datasets from its scripts.
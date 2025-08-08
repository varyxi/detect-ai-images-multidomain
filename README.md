# Detect AI-Generated Images (Multi-Domain)

This repository provides a reproducible framework to train AI-generated image detectors and evaluate the **generalization** across different domains and generators.

## ✨ Key Features

- **Cross-Domain Evaluation**: Assess how well models detect fakes from unseen domains or generators.
- **CLIP + MLP Baseline**: Lightweight, efficient detector using CLIP vision features with a small MLP head.
- **Mixture-of-Experts (MoE)**: Experimental architecture to specialize detection across image types.
- **Open Models Benchmarking**: Compare against Hugging Face models:
  - [`NYUAD-ComNets/NYUAD_AI-generated_images_detector`](https://huggingface.co/NYUAD-ComNets/NYUAD_AI-generated_images_detector)
  - [`haywoodsloan/ai-image-detector-deploy`](https://huggingface.co/haywoodsloan/ai-image-detector-deploy)

## 🚀 Quick Start

```bash
git clone https://github.com/varyxi/detect-ai-images-multidomain
cd detect-ai-images-multidomain
pip install -r requirements.txt
```

Train a detector:
```bash
python 
python train.py --config configs/clip_mlp_defactify.yaml
```

Evaluate a model:
```bash
python eval.py --model path/to/model.pth --data path/to/test_data/
```

## 📂 Datasets

Supported datasets:
- **ArtiFact** – Set of generators, diverse domains
- **Defactify** – Modern diffusion models + real images
- **AI vs Human (Alex)** – Paired real/fake images from recent competition

Scripts provided to format and split data for training.

## 📊 Results Summary

- Models trained on one domain **struggle to generalize** to others.
- **Multi-generator training** improves robustness.
- Hugging Face detectors perform well in-domain, but drop on unseen data.
- MoE shows promise for adapting across distributions.

## 📄 Citation

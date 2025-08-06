# T2I Fairness-Utility Tradeoffs

A comprehensive evaluation framework for measuring fairness and utility tradeoffs in text-to-image (T2I) generation models.

## Overview

This tool analyzes generated images to assess both utility (how well images match text prompts) and fairness (demographic diversity in generated content).
It provides both a web interface and command-line tools for batch processing.

## Features

- **Utility Metrics**: CLIP Score for measuring text-image alignment
- **Fairness Metrics**: Normalized entropy and KL divergence for demographic diversity
- **Interactive Web Interface**: Streamlit-based dashboard with Pareto charts
- **Batch Processing**: Command-line interface for large-scale evaluation
- **Multi-Model Support**: Compatible with various T2I models (Stable Diffusion, FLUX, etc.)

## Quick Start

### Setup
1. Copy the example environment file and configure your API keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Web Interface
```bash
streamlit run app.py
```

The web interface supports multiple input methods:
- **Upload ZIP files** containing images
- **Select local folders** with images
- **Upload existing JSON results** from previous `main_local.py` runs
- **Compare multiple configurations** side-by-side with Pareto charts

### Command Line
```bash
# Basic evaluation with CLIP score
python main_local.py --folder ./images --metrics clip_score

# Process images from a ZIP file
python main_local.py --folder ./images.zip --metrics clip_score

# Full evaluation with fairness metrics using OpenAI
python main_local.py --folder ./images --metrics clip_score entropy_fairness kl_fairness --api-key YOUR_OPENAI_KEY --model-provider openai

# Full evaluation with fairness metrics using Gemini
python main_local.py --folder ./images --metrics clip_score entropy_fairness kl_fairness --api-key YOUR_GEMINI_KEY --model-provider gemini

python main_local.py --folder ./images --metrics all --api-key YOUR_GEMINI_KEY --model-provider gemini

# Save results to custom location (can be loaded later in web interface)
python main_local.py --folder ./images --metrics clip_score --output ./my_results.json
```

## Metrics

- **CLIP Score**: Measures semantic similarity between generated images and text prompts
- **Normalized Entropy**: Higher values indicate more demographic diversity (fairer)
- **KL Divergence**: Lower values indicate closer to uniform demographic distribution (fairer)

## Requirements

- Python 3.8+
- PyTorch
- CLIP (for utility metrics)
- API access for fairness metrics (choose one):
  - OpenAI API
  - Gemini API
  - OpenRouter API

## Example Results

The `examples/` directory contains sample evaluation results from different T2I models, demonstrating various fairness-utility tradeoffs.

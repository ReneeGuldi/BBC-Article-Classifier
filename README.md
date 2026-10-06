# BBC-Article-Classifier

A PyTorch neural network that classifies BBC news articles into five topics using Doc2Vec embeddings, reaching 95.51% test accuracy and placing 2nd on the class leaderboard.

## Overview

This project has two parts:
1. **Neural Network Fundamentals:** A three-layer network with fixed weights, visualized as 3D plots to compare a sigmoid activation against a purely linear output.
2. **Article Classification:** 2,225 BBC news articles (business, entertainment, politics, sport, and tech) are converted into 250-dimension vectors with Gensim's Doc2Vec and classified by a feed-forward network (250 -> 128 -> 64 -> 5, ReLU, dropout 0.3, Adam, cross-entropy loss).

As a bonus, I implemented TF-IDF from scratch in NumPy and ran it through the same network for comparison.

## Results

On a held-out test of 445 articles (80/20 split):

| Text Representation | Vector Size | Test Accuracy |
| --- | --- | --- |
| Doc2Vec | 250 | **95.51%** |
| TF-IDF (from scratch) | 29,088 | 92.81% |

Tuning raised Doc2Vec accuracy from 85.17% to 95.51% (+10.34 points).

## Tech Stack

Python ·  PyTorch · Gensim · NumPy · pandas · Matplotlib

## Dataset

2,225 articles from the BBC News dataset across five categories, provided as part of the course materials.

## How to Run

The notebook was built in Google Colab. To run it:

1. Open `Homework3_Guldi_Renee.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter.
2. Run all cells. The notebook downloads the dataset and helper file automatically using `gdown`.
3. If running locally, install dependencies first: `pip install torch gensim pandas numpy matplotlib gdown`

## What I Learned

- **Text has to become numbers before a network can use it.** A neural network can't multiply a word by a weight, so the most important design decision was how to represent each article. Doc2Vec turned every article into a 250-number vector that captures meaning and context, not just word counts.
- **Representation matters more than size.** My from-scratch TF-IDF produced vectors more than 100 times larger (29,088 dimensions) yet scored lower (92.81% vs. 95.51%). A compact vector that understands context beat a huge one that only counts words.
- **Small tuning changes can have a big impact.** My first model reached 85.17%. Increasing Doc2Vec's vector size (50 -> 250) and training epochs (20 -> 100) raised it to 95.51%. Not every change helped: switching from Adam to SGD dropped accuracy to 46%, so i reverted it. Testing one change at a time showed me which changes actually mattered.
- **Activation functions are what make networks non-linear.** In Part 1, the same network with a sigmoid activation produced a curved decision boundary between 0 and 1, while removing it produced a flat, unbounded plane that could only separate the data with a straight line.

## Files

- `Homework3_Guldi_Renee.ipynb`: Part 1 visualizations, Doc2Vec pipeline, FNN training, and TF-IDF bonus
- `Guldi_Renee-HW3_Presentation.ppyx`: Project presentation slides.

## Acknowledgments

`data_processor.py` (train/test splitting) and the dataset were provided as part of CSCI-460 course materials.

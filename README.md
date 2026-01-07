# Neural Machine Translation from Scratch (PyTorch)

A minimal, from-scratch implementation of a **Neural Machine Translation (NMT) system**
built to understand how encoder–decoder models work under the hood.

---

## Seq2Seq Translation Overview

The model follows a classic **encoder–decoder** pipeline:

<img width="720" alt="seq2seq-diagram" src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgS2Pn-Q-XgrMZh8rmLcobWSg20VqdOJrxFyOv6jBfef6ojgbkvGTo7g0nSZ4nym8TXq2K4nIj-H-6k7YnfcXBAmE9oA1DpbJMoBNB9cgrywPd4ZVfpeP_gL4UTRdfLVkPlZBSmIUPtjmU/s1600/unnamed.gif" />

- The **encoder** processes the full source sentence
- The **decoder** generates the target sentence token by token
- Generation stops when the `<EOS>` token is produced

No pretrained models or external NMT libraries are used.

---

## Model Architecture

<img width="720" alt="encoder-decoder-detail" src="[https://github.com/user-attachments/assets/2f91b4a1-encoder-decoder](https://av-eks-lekhak.s3.amazonaws.com/media/__sized__/article_images/image_Z3Jneoa-thumbnail_webp-600x300.webp)" />

- **Encoder**
  - Embedding layer
  - RNN (GRU / LSTM)
  - Final hidden state represents the source sentence
- **Decoder**
  - Embedding layer
  - RNN
  - Linear + softmax output layer
  - Autoregressive decoding

Teacher forcing is applied during training.

---

## Training & Inference

- Custom dataset and DataLoader
- Padding and masking
- Cross-entropy loss
- Manual training loop
- Greedy decoding at inference time

The focus is clarity and transparency rather than performance.

---

## Running the Model

```bash
pip install torch
python train.py
python inference.py

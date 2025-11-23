# Named Entity Recognition (NER) with BiLSTM-CNN

This project implements a **Named Entity Recognition (NER)** system using a **hybrid BiLSTM-CNN model** with **word-level** and **character-level embeddings**. The model is trained on an **extended CoNLL-2003 English dataset**.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [References](#references)
- [License](#license)

---

## Introduction
Named Entity Recognition (NER) identifies entities such as **person names, locations, organizations, dates, etc.** in text. This project uses a **BiLSTM with CNN-based character embeddings** to capture both word-level context and subword features, enabling robust entity recognition, including unseen words.

---

## Dataset
The model uses an **extended CoNLL-2003 dataset**:

- **Training sentences:** 15,020 (original + additional files)  
- **Test sentences:** 7,150 (`eng.testa` + `eng.testb`)  

Each word is labeled with its NER tag (e.g., `B-PER`, `I-LOC`, `O`).  

**Input shapes for the model:**  

| Data | Shape |
|------|-------|
| X_train | (15020, 113) |
| y_train | (15020, 113, 10) |
| X_train_char | (15020, 113, 12) |
| X_test_char | (7150, 113, 12) |

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/ner-bilstm-cnn.git
cd ner-bilstm-cnn
pip install -r requirements.txt

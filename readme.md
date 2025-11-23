# Named Entity Recognition (NER) with Bi-LSTM & Character CNN

This project implements a deep learning model for **Named Entity Recognition (NER)** using **TensorFlow** and **Keras**.  
The architecture is a hybrid model that combines:

- **Word-level embeddings**
- **Character-level embeddings** processed through **Convolutional Neural Networks (CNNs)**
- A **Bidirectional LSTM (Bi-LSTM)** network for sequence modeling

This combination enables the model to effectively capture both word semantics and sub-word character patterns, improving performance on entity recognition tasks.

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


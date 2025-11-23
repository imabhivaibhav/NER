# Named Entity Recognition (NER) with Bi-LSTM & Character CNN

This project implements a deep learning model for **Named Entity Recognition (NER)** using **TensorFlow** and **Keras**.  
The architecture is a hybrid model that combines:

- **Word-level embeddings**
- **Character-level embeddings** processed through **Convolutional Neural Networks (CNNs)**
- A **Bidirectional LSTM (Bi-LSTM)** network for sequence modeling

This combination enables the model to effectively capture both word semantics and sub-word character patterns, improving performance on entity recognition tasks.

---

## 🧠 Model Architecture

The model uses a **hierarchical architecture** that captures both semantic meaning (word-level features) and morphological patterns (character-level features):

---

### 🔹 **1. Input Layer (Words)**
- Accepts **word indices**.
- Passed through a **Word Embedding layer** with:
  - **Embedding dimension:** 80

---

### 🔹 **2. Input Layer (Characters)**
- Accepts **character indices for each word**.
- Passed through a **Character Embedding layer** with:
  - **Embedding dimension:** 16
- Further processed through:
  - **TimeDistributed Conv1D (CNN):** extracts sub-word patterns such as prefixes, suffixes, and capitalization cues.
  - **GlobalMaxPooling1D:** reduces the CNN output into a compact character-level representation.

---

### 🔹 **3. Concatenation Layer**
- The **word embedding** and **character-derived features** are concatenated to form a hybrid word representation.

---

### 🔹 **4. Context Encoder**
- A **Bidirectional LSTM (48 units)** processes the concatenated vectors.
- Captures **contextual information** from both forward and backward directions.

---

### 🔹 **5. Output Layer**
- A **TimeDistributed Dense layer** with **Softmax activation**.
- Predicts the **NER tag** for each word in the sentence.

---


## 📂 Dataset Requirements

This project uses the **CoNLL-2003** dataset for training, validation, and testing.

You must have the following files available locally:

- **eng.train** — Training dataset  
- **eng.testa** — Validation dataset  
- **eng.testb** — Test dataset  

---

## 📄 Data Format (CoNLL)

Each file should follow the **standard CoNLL-2003 format**, where:

- Every **line contains one word**, its POS tag, chunk tag, and NER tag.
- **Sentences are separated** by an empty line.

Example:

```
EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
to TO B-VP O
boycott VB I-VP O
British JJ B-NP B-MISC
lamb NN I-NP O
. . O O
```




Structure per line:

```
WORD POS-TAG CHUNK-TAG NER-TAG
```


This format ensures compatibility with the Bi-LSTM + Char-CNN NER pipeline.

## 🛠️ Dependencies

Install the required Python libraries using `pip`:

```bash
pip install numpy tensorflow
```
### 📌 Requirements

- **Python:** 3.7+  
- **TensorFlow:** 2.x  
- **NumPy**








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

```python
import numpy as np

def hello():
    print("Hello World")
```

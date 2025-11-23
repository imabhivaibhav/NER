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

This project uses the **CoNLL-2003** dataset for training, validation, and testing. And modified Train data.

You must have the following files available locally:

- **eng.train** — Training dataset  
- **eng.testa** — Validation dataset  
- **eng.testb** — Test dataset  

---


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

## 🚀 Usage

### 1. Configure Paths
Open the Python script and locate the **file path section** at the top.  
Update the paths to point to your local dataset files:

- `eng.train` — Training data  
- `eng.testa` — Validation data  
- `eng.testb` — Test data

train_file = r"/path/to/your/eng.train"
testa_file = r"/path/to/your/eng.testa"
testb_file = r"/path/to/your/eng.testb"

### 2. Run the script
```
python ner_model.py

```

### 3. Process:

The script will parse the CoNLL files.

It will build the vocabulary for words, tags, and characters.

The model will train for 10 epochs (by default).

Finally, it will run a prediction on a sample custom paragraph.

## ⚙️ Hyperparameters

| Hyperparameter   | Value | Description                          |
|-----------------|-------|--------------------------------------|
| Word Embed Dim  | 80    | Dimension of word vectors            |
| Char Embed Dim  | 16    | Dimension of character vectors       |
| CNN Filters     | 20    | Number of filters for char feature extraction |
| Kernel Size     | 3     | Size of the CNN window               |
| LSTM Units      | 48    | Hidden units in the Bidirectional LSTM |
| Batch Size      | 32    | Samples per gradient update          |
| Optimizer       | Adam  | Adaptive learning rate optimizer     |

## 📝 Example Output

After training, the script predicts entities for the sentence:  

*"I eat apple every day, I leave in India, I leave in Mumbai."*

### Sample Console Output

```
Sentence 1: [('I', 'O'), ('eat', 'O'), ('apple', 'O'), ('every', 'O'), ('day', 'O'), (',', 'O')]
Sentence 2: [('I', 'O'), ('leave', 'O'), ('in', 'O'), ('India', 'I-LOC'), (',', 'O')]
Sentence 3: [('I', 'O'), ('leave', 'O'), ('in', 'O'), ('Mumbai', 'I-LOC'), ('.', 'O')]

```
## Download

[Download CoNLL-2003 Dataset (Original)](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion)

[Download CoNLL-2003 Dataset (Modified - 15020 sentences)](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion)


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

## Preprocessing

1. **Tokenization:**  
   Sentences are split into words; words are split into characters.

2. **Vocabulary construction:**  
   - Word vocabulary includes `PAD` and `UNK`.  
   - Character vocabulary includes `PAD` and `UNK`.

3. **Padding:**  
   - Sentences are padded to 113 words.  
   - Words are padded to 12 characters.

4. **Tag encoding:**  
   Tags are converted to indices and one-hot encoded for softmax classification.

**Example code snippet for converting sentences to character indices:**

```python
def sentences_to_char_indices(sentences, max_len, max_word_len):
    X_char = []
    for sent in sentences:
        sent_chars = []
        for word in sent[:max_len]:
            word_chars = [char2idx.get(c, char2idx["UNK"]) for c in word[:max_word_len]]
            word_chars += [char2idx["PAD"]] * (max_word_len - len(word_chars))
            sent_chars.append(word_chars)
        sent_chars += [[char2idx["PAD"]] * max_word_len] * (max_len - len(sent_chars))
        X_char.append(sent_chars)
    return np.array(X_char)



✅ Key fixes:  
- Heading uses `##` instead of underlines.  
- Lists properly indented with `-` for subpoints.  
- Blank lines before code blocks.  
- Code block wrapped in triple backticks with `python` for syntax highlighting.  

This will render properly as Markdown on GitHub.  

If you want, I can **fix the full README.md from that point onward**, so the rest of your file is fully GitHub-ready Markdown in the **same single file**, with no “normal text” anywhere. Do you want me to do that?

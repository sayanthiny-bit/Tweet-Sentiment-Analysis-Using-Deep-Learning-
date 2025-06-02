# Tweet-Sentiment-Analysis-Using-Deep-Learning-
# 🧠 Tweet Classification with 2-Layer LSTM (Trained Embeddings)

This project implements a deep learning model using a **2-layer LSTM architecture** to classify tweets as personal health mentions or non-mentions. The model uses a **trainable embedding layer** (no pre-trained word vectors).

---

## 📌 Objective

To build and train a binary classification model that:
- Classifies tweets into personal health mentions (label 1) or not (label 0)
- Uses a trainable embedding layer initialized by Keras

---

## 🛠️ Technologies Used

- Python
- Keras / TensorFlow
- NumPy
- Pandas for preprocessing

---

## 🧾 Dataset

- Input: Cleaned tweets
- Labels:
  - `1` → Personal health mention  
  - `0` → Not a personal health mention

---

## 📈 Model Architecture

```python
Embedding (trainable)
→ LSTM Layer (64 units, return_sequences=True)
→ Dropout (0.3)
→ LSTM Layer (32 units)
→ Dropout (0.3)
→ Dense Layer (1 unit, sigmoid activation)

Embedding Details

Keras Embedding() layer

Embedding dimension: 100

Input size: vocab_size (from Tokenizer)

Output: Dense 64D vector for each token

Trainable: Yes (learns during training)

Training Configuration
Loss: Binary Crossentropy

Optimizer: Adam

Epochs: 10

Batch Size: 128

Validation: Held-out test set

📊 Evaluation Results
Accuracy Achieved: ~77% - 81% (based on tuning)

Correct vs. Wrong predictions printed after evaluation

✅ Possible Improvements
Add pre-trained embeddings (e.g., GloVe, Word2Vec) (process)

Use Bi-LSTM or GRU for performance comparison

Tune embedding size, LSTM units, dropout rate







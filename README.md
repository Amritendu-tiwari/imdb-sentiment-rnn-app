# imdb-sentiment-rnn-app
This project is a Sentiment Analysis web app for IMDB movie reviews, built using a Simple Recurrent Neural Network (SimpleRNN) with a ReLU activation function. The app allows users to enter any movie review text, and it predicts whether the sentiment is positive or negative, along with a confidence score. 
# 🎥 IMDB Movie Review Sentiment Analysis (SimpleRNN + Streamlit)

This project is a **web-based sentiment analysis tool** for IMDB movie reviews. It uses a **Simple Recurrent Neural Network (SimpleRNN)** model with ReLU activation, trained on the IMDB dataset using Keras and TensorFlow.

## ✨ Features

- Accepts custom movie review text input
- Predicts if the review is **Positive** or **Negative**
- Displays a confidence score (probability)
- User-friendly web interface built with Streamlit

---

## 🚀 Demo

![App Screenshot](https://user-images.githubusercontent.com/your-screenshot.png)

---

## 📄 Model Details

- **Architecture**: Embedding layer → SimpleRNN (ReLU) → Dense (sigmoid)
- **Dataset**: IMDB movie reviews (50,000 reviews)
- **Vocabulary size**: Top 10,000 most frequent words
- **Max sequence length**: 500 words

---

## 💻 Installation

### Clone the repo

```bash
git clone https://github.com/yourusername/imdb-sentiment-rnn.git
cd imdb-sentiment-rnn


from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from PyPDF2 import PdfReader
import torch
import numpy as np
from collections import Counter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk

def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
nltk.download('punkt_tab')
setup_nltk()

from nltk.tokenize import word_tokenize
import pickle
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['MODEL_FOLDER'] = 'models'
app.config['VOCAB_FILE'] = 'vocab.pkl'
app.config['TRAINING_STATUS_FILE'] = 'training_status.txt'

# Initialize models and variables
lstm_model = None
vocab = None
corpus = None
word_pairs = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
is_training = False

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=150):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(hidden.squeeze(0))
        return output

def load_models():
    global lstm_model, vocab
    
    # Load vocab if exists
    vocab_path = os.path.join(app.config['MODEL_FOLDER'], app.config['VOCAB_FILE'])
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    
    # Initialize LSTM model (will be trained when PDF is uploaded)
    if vocab:
        lstm_model = LSTMModel(len(vocab)).to(device)
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'lstm_model.pt')
        if os.path.exists(model_path):
            lstm_model.load_state_dict(torch.load(model_path, map_location=device))
            lstm_model.eval()

def update_training_status(status):
    with open(os.path.join(app.config['MODEL_FOLDER'], app.config['TRAINING_STATUS_FILE']), 'w') as f:
        f.write(status)

def process_pdf(filepath):
    global corpus, word_pairs, vocab, lstm_model, is_training
    
    is_training = True
    update_training_status("Processing PDF...")
    
    # Read PDF text
    text = ""
    try:
        with open(filepath, 'rb') as f:
            reader = PdfReader(f)
            num_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
                update_training_status(f"Processing page {i+1}/{num_pages}...")
    except Exception as e:
        is_training = False
        update_training_status("Error processing PDF")
        print(f"Error reading PDF: {e}")
        raise
    
    # Split text into sentences
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
    corpus = sentences
    
    # Tokenize and build vocabulary
    update_training_status("Building vocabulary...")
    tokens =    word_tokenize(text.lower())
    vocab = {'<unk>': 0, '<pad>': 1}
    
    # Update vocabulary with new tokens
    for token in Counter(tokens).keys():
        if token not in vocab:
            vocab[token] = len(vocab)
    
    # Save vocabulary
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    with open(os.path.join(app.config['MODEL_FOLDER'], app.config['VOCAB_FILE']), 'wb') as f:
        pickle.dump(vocab, f)
    
    # Build word transition pairs
    update_training_status("Building word transitions...")
    word_pairs = {}
    for sentence in corpus:
        words = word_tokenize(sentence.lower())
        for i in range(len(words)-1):
            current = words[i]
            next_word = words[i+1]
            if current not in word_pairs:
                word_pairs[current] = []
            word_pairs[current].append(next_word)
    
    # Prepare training data for LSTM
    update_training_status("Preparing training data...")
    input_sentences = text.split('\n')
    input_numerical_sentences = []
    
    for sentence in input_sentences:
        tokens = word_tokenize(sentence.lower())
        numerical = [vocab.get(token, vocab['<unk>']) for token in tokens]
        input_numerical_sentences.append(numerical)
    
    # Create training sequences
    training_sequence = []
    for sentence in input_numerical_sentences:
        for i in range(1, len(sentence)):
            training_sequence.append(sentence[:i+1])
    
    # Find max sequence length
    max_len = max(len(seq) for seq in training_sequence) if training_sequence else 0
    
    # Pad sequences
    padded_sequences = []
    targets = []
    for seq in training_sequence:
        padded_seq = [vocab['<pad>']] * (max_len - len(seq)) + seq
        padded_sequences.append(padded_seq[:-1])  # Input is all words except last
        targets.append(seq[-1])  # Target is last word
    
    # Convert to tensors
    X = torch.tensor(padded_sequences, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)
    
    # Create dataset and dataloader
    dataset = CustomDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train LSTM model
    if len(vocab) > 1:  # Only train if we have more than just <unk> and <pad>
        lstm_model = train_lstm_model(vocab, dataloader)
    
    is_training = False
    update_training_status("Ready")

def train_lstm_model(vocab, dataloader, epochs=10, learning_rate=0.001):
    model = LSTMModel(len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training LSTM model...")
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        update_training_status(f"Training epoch {epoch+1}/{epochs}...")
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), os.path.join(app.config['MODEL_FOLDER'], 'lstm_model.pt'))
    model.eval()
    return model

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def text_to_indices(text, vocab):
    tokens = word_tokenize(text.lower())
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

def predict_next_word(text, top_k=3):
    global lstm_model, vocab, word_pairs
    
    suggestions = []
    words = word_tokenize(text.lower())
    
    if not words:
        return []
    
    # 1. First try word transition pairs
    if word_pairs:
        last_word = words[-1]
        if last_word in word_pairs:
            transition_suggestions = Counter(word_pairs[last_word]).most_common(top_k)
            suggestions.extend([word for word, _ in transition_suggestions])
    
    # 2. Try LSTM prediction if we have a trained model
    if lstm_model and vocab and len(suggestions) < top_k:
        try:
            numerical = text_to_indices(text, vocab)
            max_len = lstm_model.lstm.input_size if hasattr(lstm_model.lstm, 'input_size') else 100
            padded = [vocab['<pad>']] * (max_len - len(numerical)) + numerical
            input_tensor = torch.tensor([padded], dtype=torch.long).to(device)
            
            with torch.no_grad():
                output = lstm_model(input_tensor)
                probs = torch.softmax(output, dim=1)
                top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
            
            # Convert indices to words
            idx_to_word = {v: k for k, v in vocab.items()}
            lstm_suggestions = [idx_to_word[idx.item()] for idx in top_indices[0]]
            suggestions.extend(lstm_suggestions)
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
    
    # Remove duplicates and return top_k suggestions
    unique_suggestions = []
    seen = set()
    for word in suggestions:
        if word not in seen and word != '<unk>' and word != '<pad>':
            seen.add(word)
            unique_suggestions.append(word)
            if len(unique_suggestions) >= top_k:
                break
    
    return unique_suggestions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.endswith('.pdf'):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.pdf')
            file.save(filepath)
            
            # Start processing in background
            from threading import Thread
            Thread(target=process_pdf, args=(filepath,)).start()
            
            return redirect(url_for('loading'))
        
        return redirect(request.url)
    except Exception as e:
        print(f"Error in upload: {e}")
        return "Error processing PDF. Please try another file.", 500

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/training_status')
def training_status():
    try:
        status_path = os.path.join(app.config['MODEL_FOLDER'], app.config['TRAINING_STATUS_FILE'])
        if os.path.exists(status_path):
            with open(status_path, 'r') as f:
                status = f.read()
                print(status)
                if status == "Ready":
                    return jsonify({"status": status, "redirect": True})
                return jsonify({"status": status, "redirect": False})
        return jsonify({"status": "Starting...", "redirect": False})
    except Exception as e:
        return jsonify({"status": f"Error: {str(e)}", "redirect": False})

@app.route('/write')
def write():
    if not vocab or is_training:
        return redirect(url_for('index'))
    return render_template('write.html')

@app.route('/suggest')
def suggest():
    text = request.args.get('text', '').strip()
    if not text:
        return jsonify([])
    
    try:
        suggestions = predict_next_word(text, top_k=3)
        return jsonify(suggestions)
    except Exception as e:
        print(f"Error in suggestion: {e}")
        return jsonify([])

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    load_models()
    app.run(debug=True)
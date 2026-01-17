# ============================================================================
# APP.PY - SISTEM EKSTRAKSI KELUHAN (AUTO-MERGE & CLEANING)
# Fitur: Menggabungkan frasa berurutan & Membuang negasi gantung
# ============================================================================

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel, pipeline
from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import os
import json
import re
from collections import Counter

# ============================================================================
# 1. KONFIGURASI & PATH
# ============================================================================

SENTIMENT_MODEL_PATH = "./model/saved_model"
LABEL_ENCODER_PATH = "./model/label_encoder.pkl"
MODEL_NAME = "indobenchmark/indobert-base-p1"
POS_MODEL_NAME = "w11wo/indonesian-roberta-base-posp-tagger"
CBD_BERT_PATH = "./model/indobert_clause_detection_model.pt"
CBD_CRF_PATH = "./model/crf_clause_detection_model.pkl"

# File Text
NEGATIVE_KEYWORDS_FILE = "./model/negative.txt" 
POSITIVE_KEYWORDS_FILE = "./model/positive.txt"
STOP_KELUHAN_FILE = "./model/badword.txt"

MAX_LEN_CBD = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ============================================================================
# 2. LOADER KAMUS
# ============================================================================

def load_keywords_set(filepath):
    s = set()
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip().lower()
                    if word: s.add(word)
            print(f"✓ Loaded {len(s)} words from {filepath}")
        except: pass
    else:
        print(f"⚠ CRITICAL: File not found: {filepath}")
    return s

ALL_NEGATIVES = load_keywords_set(NEGATIVE_KEYWORDS_FILE)
ALL_POSITIVES = load_keywords_set(POSITIVE_KEYWORDS_FILE)
STOP_WORDS = load_keywords_set(STOP_KELUHAN_FILE)

# Backup Manual
MANUAL_BACKUP_NEGATIVES = {
    "kotor", "rusak", "jelek", "bau", "mahal", "kumuh", "berantakan", "bising",
    "berisik", "panas", "gersang", "sempit", "macet", "antri", "lama", "lelet",
    "lambat", "kasar", "judes", "curang", "liar", "hancur", "bolong", "licin",
    "gelap", "suram", "sesak", "pengap", "amis", "pesing", "sampah", "pungli",
    "parah", "kecewa", "buruk", "susah", "sulit", "ribet", "semrawut", "tolol", "anjeng", "anjing"
}
ALL_NEGATIVES = ALL_NEGATIVES.union(MANUAL_BACKUP_NEGATIVES)

# Manual Positif
MANUAL_POSITIVE_FALLBACK = {
    "baik", "bagus", "indah", "cantik", "nyaman", "aman", "bersih", "ramah", 
    "luas", "terawat", "rapi", "sejuk", "dingin", "puas", "enak", "sedap",
    "mantap", "keren", "oke", "layak", "memadai", "profesional", "sigap", "halus", "sopan"
}
ALL_POSITIVES = ALL_POSITIVES.union(MANUAL_POSITIVE_FALLBACK)

# Negators & Ignore
NEGATORS = {"tidak", "kurang", "belum", "bukan", "enggak", "gak", "tak", "jangan", "anti", "susah", "sulit", "ga"}

# Kata pemutus (Stop Marking jika ketemu ini)
IGNORE_WORDS = {
    "yang", "dan", "di", "ke", "dari", "adalah", "karena", "karna", 
    "saya", "kita", "kami", "anda", "mereka", "dia", 
    "mulai", "supaya", "sebagai", "jika", "kalau",
    "semoga", "tapi", "tetapi", "namun", "walaupun"
}

key_norm = {
    "yg": "yang", "gak": "tidak", "ga": "tidak", "g": "tidak", "nggak": "tidak", "gk": "tidak",
    "kalo": "kalau", "klo": "kalau", "kl": "kalau",
    "bgt": "banget", "bg": "banget", "dgn": "dengan", "dg": "dengan",
    "krn": "karena", "karna": "karena", "tdk": "tidak", "tak": "tidak",
    "jd": "jadi", "jdi": "jadi", "bkn": "bukan", "sdh": "sudah",
    "tp": "tapi", "tpi": "tapi", "sy": "saya", "aku": "saya",
    "bgs": "bagus", "good": "bagus", "bad": "jelek",
    "d": "di", "tmpt": "tempat", "msh": "masih", "tau": "tahu",
    "ad": "ada", "sm": "sama", "dr": "dari", "utk": "untuk"
}

# ============================================================================
# 3. LOADER AI MODELS
# ============================================================================
def clean_text_user(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    fixed = [key_norm.get(w, w) for w in words]
    return ' '.join(fixed)

print("--- Initializing Models ---")

try: tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except: tokenizer = None

label_encoder = None
try:
    if os.path.exists(LABEL_ENCODER_PATH):
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    else:
        class DummyEnc:
            def __init__(self): self.classes_ = ['negatif', 'netral', 'positif']
            def inverse_transform(self, idx): return [self.classes_[idx[0]]]
        label_encoder = DummyEnc()
except: pass

sentiment_model = None
try:
    num_labels = len(label_encoder.classes_) if label_encoder else 3
    sentiment_model = BertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH, num_labels=num_labels)
    sentiment_model.to(device)
    sentiment_model.eval()
    print("✓ Sentiment Model Loaded")
except: sentiment_model = None

pos_pipeline = None
try:
    pos_pipeline = pipeline("token-classification", model=POS_MODEL_NAME, tokenizer=POS_MODEL_NAME, aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
    print("✓ POS Tagger Loaded")
except: pass

class IndoBERT_FineTune(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    def get_features(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state)

cbd_bert_model, cbd_crf_model = None, None
try:
    import sklearn_crfsuite
    if os.path.exists(CBD_BERT_PATH) and os.path.exists(CBD_CRF_PATH):
        cbd_bert_model = IndoBERT_FineTune(MODEL_NAME, num_labels=3)
        cbd_bert_model.load_state_dict(torch.load(CBD_BERT_PATH, map_location=device))
        cbd_bert_model.to(device)
        cbd_bert_model.eval()
        with open(CBD_CRF_PATH, 'rb') as f: cbd_crf_model = pickle.load(f)
        print("✓ CBD Model Loaded")
except: pass


# ============================================================================
# 4. CORE FUNCTIONS
# ============================================================================

def detect_clauses(text):
    clean_text = re.sub(r'\s+', ' ', text).strip()
    if not clean_text: return []
    if not cbd_bert_model or not cbd_crf_model: return re.split(r'[.!?,;]\s*', clean_text)
    
    def extract_bert_features(tokens):
        encoded = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding='max_length', max_length=128).to(device)
        with torch.no_grad():
            f = cbd_bert_model.get_features(encoded['input_ids'], encoded['attention_mask'])
            return f[0][:len(tokens)].cpu().numpy()

    rough_splits = re.split(r'([.!?;])', clean_text)
    final_clauses = []
    for segment in rough_splits:
        segment = segment.strip()
        if not segment or segment in ['.', '!', '?', ';']: continue
        raw_tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", segment)
        if not raw_tokens: continue
        try:
            feats = extract_bert_features(raw_tokens)
            tags = cbd_crf_model.predict_single(feats)
            curr = []
            for tok, tag in zip(raw_tokens, tags):
                if tag == "B-CLAUSE":
                    if curr: final_clauses.append(" ".join(curr))
                    curr = [tok]
                else: curr.append(tok)
            if curr: final_clauses.append(" ".join(curr))
        except: final_clauses.append(segment)
    return [c.strip() for c in final_clauses if len(c.strip()) > 3]

def predict_sentiment_global(text):
    if not sentiment_model: return "non-negatif" # Default aman
    try:
        inputs = tokenizer(clean_text_user(text), return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
        
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        original_label = label_encoder.inverse_transform([pred_id])[0] # Hasil asli: positif/netral/negatif
        
        # --- MAPPING BARU: 2 KELAS SAJA ---
        if original_label.lower() == 'negatif':
            return 'negatif'
        else:
            # Positif dan Netral digabung jadi 'non-negatif'
            return 'non-negatif'
            
    except: return "non-negatif"

# --- FUNGSI DARURAT (RESCUE) ---
def rescue_extraction_window(text):
    words = text.split()
    results = []
    for i, w in enumerate(words):
        w_clean = re.sub(r'[^a-z0-9]', '', w)
        if w_clean in ALL_NEGATIVES:
            start = max(0, i - 2)
            end = min(len(words), i + 2)
            phrase_words = words[start:end]
            if phrase_words[0] in IGNORE_WORDS: phrase_words.pop(0)
            if phrase_words and phrase_words[-1] in IGNORE_WORDS: phrase_words.pop(-1)
            if phrase_words: results.append(" ".join(phrase_words))
    return results

# --- HELPER: MAP LABEL POS ---
def map_pos_label(raw_label):
    clean_label = raw_label.replace("B-", "").replace("I-", "")
    if clean_label in ["NNO", "NNP", "PRN", "PRK"]: return "NNO" # Noun
    if clean_label in ["ADJ", "ADK"]: return "ADJ" # Adjective
    if clean_label in ["NEG"]: return "NEGATOR" # Negation
    if clean_label in ["KUA", "NUM", "ART"]: return "DET" # Quantifier
    if clean_label in ["VBI", "VBT", "VBP", "VBL", "VBE"]: return "VERB" # Verb
    return "OTHER"

# --- ALGORITMA EKSTRAKSI: BOOLEAN MASKING (AUTO MERGE) ---
def extract_complaints_user_algo(raw_text):
    norm_text = clean_text_user(raw_text)
    
    if not pos_pipeline: return rescue_extraction_window(norm_text)
    try: 
        pos_results = pos_pipeline(norm_text)
    except: 
        return rescue_extraction_window(norm_text)

    if not pos_results: return rescue_extraction_window(norm_text)

    tokens = []
    try:
        current_word = pos_results[0]['word']
        current_label = pos_results[0]['entity_group']
        current_end = pos_results[0]['end']
        
        for i in range(1, len(pos_results)):
            t = pos_results[i]
            if t['start'] == current_end:
                current_word += t['word'].replace('##', '')
                current_end = t['end']
            else:
                tokens.append({'word': current_word, 'label': current_label})
                current_word = t['word']
                current_label = t['entity_group']
                current_end = t['end']
        tokens.append({'word': current_word, 'label': current_label})
    except:
        return rescue_extraction_window(norm_text)

    # 2. Labeling Token
    labeled_tokens = []
    for t in tokens:
        word = re.sub(r'[^a-z0-9]', '', t['word'].lower().strip())
        if not word: continue
        
        pos_label = map_pos_label(t['label'])
        
        # Priority Mapping
        if word in NEGATORS: final_label = "NEGATOR"
        elif word in ALL_NEGATIVES: final_label = "NEG_ADJ"
        elif word in ALL_POSITIVES: final_label = "POS_ADJ"
        else: final_label = pos_label
        
        labeled_tokens.append({'word': word, 'label': final_label})

    # 3. LOGIKA MERGING (GANDENG)
    # Kita gunakan array boolean untuk menandai kata mana saja yang terpilih.
    # Kata yang bersebelahan (indeks berurutan) akan otomatis jadi satu kalimat.
    
    selected_indices = [False] * len(labeled_tokens)

    for i, token in enumerate(labeled_tokens):
        
        # --- RULE 1: KATA NEGATIF (Kotor, Mahal, Rusak) ---
        if token['label'] == "NEG_ADJ":
            selected_indices[i] = True # Mark kata ini
            
            # Expand Kiri
            curr = i - 1
            while curr >= 0:
                w_prev = labeled_tokens[curr]['word']
                l_prev = labeled_tokens[curr]['label']
                if w_prev in IGNORE_WORDS or w_prev in STOP_WORDS: break
                
                # Terima Noun, Det, Negator, atau Adjective lain
                if l_prev in ["NNO", "DET", "NEGATOR", "POS_ADJ", "ADJ"]:
                    selected_indices[curr] = True
                    curr -= 1
                else: break
            
            # Expand Kanan
            curr = i + 1
            while curr < len(labeled_tokens):
                w_next = labeled_tokens[curr]['word']
                l_next = labeled_tokens[curr]['label']
                if w_next in IGNORE_WORDS or w_next in STOP_WORDS: break
                
                if l_next in ["NNO", "DET"]:
                    selected_indices[curr] = True
                    curr += 1
                else: break

        # --- RULE 2: NEGATOR (Kurang/Tidak/Bukan) ---
        # SYARAT: HARUS diikuti kata sifat/kerja yang valid.
        elif token['label'] == "NEGATOR":
            if i + 1 < len(labeled_tokens):
                next_token = labeled_tokens[i+1]
                
                # Cek kelayakan kata depannya
                is_valid_next = (
                    next_token['label'] in ["POS_ADJ", "ADJ", "NEG_ADJ", "VERB"] or 
                    (next_token['word'] not in STOP_WORDS and next_token['word'] not in IGNORE_WORDS)
                )

                if is_valid_next:
                    selected_indices[i] = True     # Mark 'Kurang'
                    selected_indices[i+1] = True   # Mark 'Terawat'
                    
                    # Expand Kiri (Cari Subjek)
                    curr = i - 1
                    while curr >= 0:
                        w_prev = labeled_tokens[curr]['word']
                        l_prev = labeled_tokens[curr]['label']
                        if w_prev in IGNORE_WORDS or w_prev in STOP_WORDS: break
                        if l_prev in ["NNO", "DET"]:
                            selected_indices[curr] = True
                            curr -= 1
                        else: break

    # 4. MEMBANGUN FRASA (Merging)
    # Gabungkan kata-kata yang index-nya bernilai True secara berurutan
    final_phrases = []
    current_phrase = []

    for i in range(len(labeled_tokens)):
        if selected_indices[i]:
            current_phrase.append(labeled_tokens[i]['word'])
        else:
            if current_phrase:
                final_phrases.append(" ".join(current_phrase))
                current_phrase = []
    
    # Jangan lupa sisa terakhir
    if current_phrase:
        final_phrases.append(" ".join(current_phrase))

    # 5. FINAL FILTERING (Hapus Negator Gantung)
    cleaned_list = []
    seen = set()
    
    for p in final_phrases:
        words = p.split()
        # Jika frasa cuma 1 kata, dan kata itu NEGATOR -> Buang (Contoh: "kurang")
        if len(words) == 1 and words[0] in NEGATORS:
            continue
        
        # Jika cuma 1 kata, dan itu bukan NEG_ADJ -> Hati-hati (tapi kita loloskan cacian)
        
        p = p.strip()
        if p and p not in seen:
            cleaned_list.append(p)
            seen.add(p)

    # Fail Safe
    if not cleaned_list:
        return rescue_extraction_window(norm_text)
            
    return cleaned_list

# ============================================================================
# 5. ROUTES
# ============================================================================
app = Flask(__name__)

@app.route('/')
def home(): return render_template("index.html")

@app.route('/wisata_list', methods=['GET'])
def wisata_list():
    folder = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(folder): return jsonify({'wisata': []})
    wisata_files = [f for f in os.listdir(folder) if f.endswith('.json')]
    names = [f.replace('.json','').replace('_',' ').title() for f in wisata_files]
    return jsonify({'wisata': names})

@app.route('/analisis_wisata', methods=['POST'])
def analisis_wisata():
    req = request.get_json(silent=True)
    if not req or 'wisata' not in req: return jsonify({'error': 'Missing wisata name'}), 400
    
    wisata_name = req['wisata']
    filename = wisata_name.lower().replace(' ','_') + '.json'
    filepath = os.path.join(os.path.dirname(__file__), 'data', filename)
    
    if not os.path.exists(filepath): return jsonify({'error': 'File not found'}), 404
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    reviews = data.get('reviews', [])
    
    complaint_data = []
    
    # UPDATE: Variabel Counter Baru
    count_negatif = 0
    count_non_negatif = 0
    
    all_nno_words = []
    
    for review in reviews:
        # 1. Clause Detection
        clauses = detect_clauses(review)
        for clause in clauses:
            # 2. Sentimen (Outputnya sudah pasti 'negatif' atau 'non-negatif')
            sent_label = predict_sentiment_global(clause)
            
            if sent_label == 'negatif':
                count_negatif += 1
                
                # 3. Pipeline Ekstraksi (Hanya jalan kalau Negatif)
                extracted = extract_complaints_user_algo(clause)
                if extracted:
                    for item in extracted: all_nno_words.append(item.split()[0])
                    complaint_data.append({
                        "Ulasan Negatif": clause, 
                        "Keluhan": ", ".join(extracted)
                    })
            else:
                # Ini mencakup Positif & Netral
                count_non_negatif += 1

    top_5 = [{'word': k, 'count': v} for k, v in Counter(all_nno_words).most_common(5)]
    for i, item in enumerate(complaint_data, 1): item['No'] = i

    # UPDATE JSON RESPONSE
    return jsonify({
        'wisata': wisata_name,
        'total': len(reviews), # Total klausa/kalimat yang diproses
        'non_negatif': count_non_negatif, # Gabungan Positif + Netral
        'negatif': count_negatif,         # Murni Negatif
        'results': [], 
        'complaints': complaint_data, 
        'top_nouns': top_5
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5132, debug=True)

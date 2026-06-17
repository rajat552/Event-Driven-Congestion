import os
import pickle
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer

def generate_embeddings():
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    csv_path = r"dataset/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Event dataset CSV not found at: {csv_path}")

    print(f"Loading events from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')

    # The same start_date as in prepare_astram_dataset.py
    start_date = datetime(2023, 11, 9, 0, 0, 0)
    
    doy2emb = {}
    
    print("Processing events and generating embeddings...")
    for idx, row in df.iterrows():
        start_dt = row['start_datetime']
        if pd.isna(start_dt):
            continue
            
        t_start_idx = int((start_dt.tz_localize(None) - start_date).total_seconds() // 600)
        if t_start_idx < 0:
            continue
        doy_exact = (t_start_idx // 144) % 365
        
        # T3STID queries using `int(marker_x[..., 3] * 365)`. 
        # In prepare_astram_dataset.py, marker_x[..., 3] is `doy_exact / 364.0`.
        query_key = int((doy_exact / 364.0) * 365)
        
        # Build text description
        event_type = str(row.get('event_type', ''))
        event_cause = str(row.get('event_cause', ''))
        priority = str(row.get('priority', ''))
        desc = str(row.get('description', ''))
        
        text = f"Event Type: {event_type}. Cause: {event_cause}. Priority: {priority}. Description: {desc}"
        
        emb = model.encode(text)
        
        if query_key not in doy2emb:
            doy2emb[query_key] = []
            
        # T3STID expects list of tuples: (id, type, embedding)
        doy2emb[query_key].append((row['id'], event_type, emb))

    output_path = "dataset/AstramBengaluru/event_embeddings.pickle"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(doy2emb, f)
        
    print(f"Generated embeddings for {len(doy2emb)} unique days.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_embeddings()

# sem-map

## Mapping Supplier JSON to Master JSON using FAISS & Sentence Embeddings

This document explains the Jupyter Notebook code that maps supplier data (unclean) to master data (clean) using **sentence embeddings** and **FAISS** for efficient similarity search. The goal is to find the closest matching items in the master dataset for each item in the supplier dataset.

---

## **1. Setup & Dependencies**
### Code Snippet:
```python
%pip install sentence-transformers faiss-cpu numpy
%pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Explanation:
- **Libraries Used**:
  - `sentence-transformers`: Generates embeddings for text data.
  - `faiss-cpu`: Efficient similarity search library (CPU version).
  - `numpy`: Numerical computations.
  - `torch`: PyTorch for GPU acceleration (CUDA 11.8 version).

- **Why?**  
  These libraries enable embedding generation (`sentence-transformers`) and fast nearest-neighbor search (`faiss`). PyTorch with CUDA ensures GPU acceleration if available.

---

## **2. Data Loading**
### Code Snippet:
```python
with open('./content/clean_data.json', 'r') as f:
    clean_data = json.load(f)  # Master data

with open('./content/unclean_data.json', 'r') as f:
    unclean_data = json.load(f)  # Supplier data
```

### Explanation:
- **clean_data.json**: Contains structured master items (e.g., `"120874": {name, price, manufacturer...}`).
- **unclean_data.json**: Contains supplier items (e.g., `"515278": {c_item_name, c_mfg_name...}`).
- **Goal**: Map `unclean_data` items to their closest matches in `clean_data`.

---

## **3. Model Initialization**
### Code Snippet:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L12-v2', device=device)
```

### Explanation:
- **Model Choice**: `all-MiniLM-L12-v2` is a lightweight model for generating 384-dimensional embeddings. Suitable for short text.
- **Device**: Uses GPU (`cuda`) if available for faster processing.

---

## **4. FAISS Index Creation**
### Code Snippet:
```python
clean_texts = [
    f"{item.get('name','')} {item.get('manufacturer_name', '')}"
    for item in clean_data.values()
]

# Generate embeddings for clean data
clean_embeddings = model.encode(clean_texts, convert_to_tensor=True, show_progress_bar=True)
clean_embeddings = clean_embeddings.cpu().numpy()

# Create FAISS index (L2 distance)
dimension = clean_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(clean_embeddings)
faiss.write_index(index, "faiss_index_FlatL2-n-m.bin")
```

### Explanation:
- **Text Construction**: Combines `name` and `manufacturer_name` from `clean_data` to represent each item (e.g., `"Moss Fresh Gel Eye Drop Syntho Pharmaceuticals Pvt Ltd"`).
- **Embeddings**: Convert text to 384-dimensional vectors using the SentenceTransformer model.
- **FAISS Index**: 
  - `IndexFlatL2`: Uses Euclidean distance for similarity search.
  - Saved to disk for reuse (`faiss_index_FlatL2-n-m.bin`).

---

## **5. Similarity Search for Unclean Data**
### Code Snippet:
```python
unclean_texts = [
    f"{item['c_item_name']} {item['c_item_name']} {item['c_item_name']} {item['c_mfg_name']} {item['c_mfg_name']}"
    for item in unclean_data_subset.values()
]

unclean_embeddings = model.encode(unclean_texts, convert_to_tensor=True)
distances, indices = index.search(unclean_embeddings, k=10)
```

### Explanation:
- **Text Construction for Unclean Data**:  
  Repeats `c_item_name` 3 times and `c_mfg_name` 2 times to give higher weight to the item name (e.g., `"MOSS FRESH EYE DROPS MOSS FRESH EYE DROPS MOSS FRESH EYE DROPS SYNTHO PHARMA CHEMICALS SYNTHO PHARMA CHEMICALS"`).
- **Search**: For each unclean item, find the top 10 closest matches in `clean_data` using the FAISS index.

---

## **6. Result Interpretation**
### Example Output:
```
Unclean Item: MOSS FRESH EYE DROPS -- SYNTHO PHARMA CHEMICALS
Top 10 Closest Matches:
  {"name": "Moss Fresh Gel Eye Drop", "manufacturer_name": "Syntho Pharmaceuticals Pvt Ltd"...} (Distance: 0.603)
```

### Explanation:
- **Distance**: Lower values indicate higher similarity (Euclidean distance).
- **Matches**: The top 10 master items most similar to the unclean item. For example:
  - Supplier item `MOSS FRESH EYE DROPS` maps to `Moss Fresh Gel Eye Drop` in the master data.

---

## **7. Saving Results**
### Code Snippet:
```python
output_file = "matches.json"
results = []
for item_id, unclean_item in unclean_data_subset.items():
    match_data = {
        "unclean_item": {
            "id": item_id,
            "name": unclean_item["c_item_name"],
            "manufacturer": unclean_item["c_mfg_name"]
        },
        "matches": [
            {"id": clean_id, "match": clean_item, "distance": float(dist)}
            for clean_id, clean_item, dist in zip(...)
        ]
    }
    results.append(match_data)
json.dump(results, open(output_file, "w"), indent=4)
```

### Explanation:
- **Output Format**: Saves results to `matches.json` with:
  - Unclean item details.
  - Top 10 matches from the master data (IDs, metadata, and distances).

---

## **How to Run**
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt  # Contains sentence-transformers, faiss-cpu, numpy, torch
   ```
2. **Folder Structure**:
   ```
   project/
   ├── main.ipynb
   ├── content/
   │   ├── clean_data.json    # Master data
   │   └── unclean_data.json # Supplier data
   ```
3. **Run the Notebook**: Execute cells in order. Ensure GPU is available for faster embeddings.

---

## **Key Notes**
- **Text Weighting**: Adjust the text construction logic (e.g., `f"{name} {name} {name} {mfg} {mfg}"`) to prioritize specific fields.
- **FAISS Tuning**: Experiment with `IndexFlatIP` (inner product) or `IndexHNSWFlat` (approximate search) for speed/accuracy tradeoffs.
- **Model Choice**: Larger models (e.g., `all-mpnet-base-v2`) may improve accuracy but require more resources.

By following this workflow, you can map supplier items to a standardized master catalog efficiently.
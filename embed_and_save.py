import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load data
df = pd.read_csv("Training Dataset.csv")

# Make each row a text chunk
chunks = df.apply(lambda row: row.to_json(), axis=1).tolist()

# Embed
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True)

# Save
np.save('embeddings.npy', embeddings)
df.to_csv('chunks.csv', index=False)

print("Embeddings saved.")

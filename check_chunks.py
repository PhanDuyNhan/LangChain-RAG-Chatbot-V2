import sys
sys.path.insert(0, 'src')
from ingest import load_vector_store, create_embeddings

emb = create_embeddings()
vs = load_vector_store(emb)
docs = vs.get()

print(f'Tong chunks: {len(docs["ids"])}')
for i, (doc, meta) in enumerate(zip(docs['documents'][:5], docs['metadatas'][:5])):
    print(f'\n--- Chunk {i+1} | Trang {meta.get("page","?")} ---')
    print(doc[:400])
    print('...')

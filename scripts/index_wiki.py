import asyncio
import os
import sys

# Ensure src is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

from mw_mcp_server.wiki.api_client import MediaWikiClient
from mw_mcp_server.embeddings.embedder import Embedder
from mw_mcp_server.embeddings.index import FaissIndex
from mw_mcp_server.embeddings.models import IndexedDocument

async def main():
    print("Initializing clients...")
    mw_client = MediaWikiClient()
    embedder = Embedder()
    index = FaissIndex()
    
    # 1. Fetch Pages
    print("Fetching page list...")
    # Consider pagination for large wikis (simplified here)
    titles = await mw_client.get_all_pages(limit=500)
    print(f"Found {len(titles)} pages.")
    
    documents = []
    
    # 2. Process each page
    for i, title in enumerate(titles):
        print(f"Processing ({i+1}/{len(titles)}): {title}")
        text = await mw_client.get_page_wikitext(title)
        if not text:
            continue
            
        # Very simple chunking: whole page or lines
        # For better RAG, use a recursive character splitter or similar.
        # Here we just treat the whole page as one doc if small, or truncate.
        # A simple improvement: split by paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        for p_idx, para in enumerate(paragraphs):
            # Skip very short content
            if len(para) < 50:
                continue
                
            documents.append(IndexedDocument(
                page_title=title,
                text=para,
                namespace=0
            ))
            
    if not documents:
        print("No documents to index.")
        return

    print(f"Generated {len(documents)} chunks. Creating embeddings (this may take time)...")
    texts = [d.text for d in documents]
    
    # Batch embedding (openai limit usually 2048 or so, handling batches of 10-20 is safe)
    embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Embedding batch {i}-{i+len(batch)}...")
        try:
            batch_emb = await embedder.embed(batch)
            embeddings.extend(batch_emb)
        except Exception as e:
            print(f"Error embedding batch: {e}")
            # Insert zeros or skip? Skipping avoids size mismatch issues if handled carefully
            # But index build requires len(docs) == len(embeddings)
            # Simple retry or fail logic needed. For now, failing index build.
            raise e

    print("Building Faiss index...")
    index.build(embeddings, documents)
    
    print("Saving index...")
    index.save()
    print("Done! Index updated.")

if __name__ == "__main__":
    asyncio.run(main())

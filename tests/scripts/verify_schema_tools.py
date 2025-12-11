
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("src"))

from mw_mcp_server.embeddings.index import FaissIndex
from mw_mcp_server.embeddings.models import IndexedDocument
from mw_mcp_server.tools.schema_tools import tool_get_categories, tool_get_properties, tool_list_pages

# Constants
NS_MAIN = 0
NS_CATEGORY = 14
NS_PROPERTY = 102

async def main():
    print("Initializing FaissIndex...")
    # Use memory-only index (temporary paths)
    index = FaissIndex(index_path="temp_index.bin", meta_path="temp_meta.json")
    
    # Create dummy documents
    docs = [
        # Main pages
        IndexedDocument(page_title="Main Page", text="Content...", namespace=NS_MAIN),
        IndexedDocument(page_title="Physics", text="Science...", namespace=NS_MAIN),
        
        # Categories
        IndexedDocument(page_title="Category:Science", text="Cat content", namespace=NS_CATEGORY),
        IndexedDocument(page_title="Category:Physics", text="Cat content", namespace=NS_CATEGORY),
        IndexedDocument(page_title="Category:People", text="Cat content", namespace=NS_CATEGORY),
        
        # Properties
        IndexedDocument(page_title="Property:Has author", text="Prop content", namespace=NS_PROPERTY),
        IndexedDocument(page_title="Property:Has date", text="Prop content", namespace=NS_PROPERTY),
        IndexedDocument(page_title="Property:Is part of", text="Prop content", namespace=NS_PROPERTY),
    ]
    
    # Embeddings (mocked - just random vectors)
    import numpy as np
    embeddings = np.random.rand(len(docs), 128).tolist() # 128 dim
    
    print("Adding documents...")
    index.add_documents(docs, embeddings)
    
    # 1. Test get_pages_by_namespace
    print("\n--- Testing get_pages_by_namespace ---")
    cats = index.get_pages_by_namespace(NS_CATEGORY)
    print(f"Categories (all): {cats}")
    assert "Category:Physics" in cats
    assert "Property:Has author" not in cats
    
    props = index.get_pages_by_namespace(NS_PROPERTY)
    print(f"Properties (all): {props}")
    assert "Property:Has author" in props
    
    # 2. Test tool_get_categories (Search/Prefix)
    print("\n--- Testing tool_get_categories (Prefix) ---")
    phys_cats = await tool_get_categories(index, prefix="Phys")
    print(f"Categories (Phys*): {phys_cats}")
    assert "Category:Physics" in phys_cats
    assert "Category:People" not in phys_cats
    
    # 3. Test tool_get_categories (Validation)
    print("\n--- Testing tool_get_categories (Validation) ---")
    valid_check = await tool_get_categories(index, names=["Science", "InvalidCat"])
    print(f"Categories (Validation): {valid_check}")
    assert "Category:Science" in valid_check
    assert "Category:InvalidCat" not in valid_check
    
    # 4. Test tool_get_properties
    print("\n--- Testing tool_get_properties (Prefix) ---")
    has_props = await tool_get_properties(index, prefix="Has")
    print(f"Properties (Has*): {has_props}")
    assert "Property:Has author" in has_props
    assert "Property:Is part of" not in has_props
    
    # 4b. Test tool_list_pages (Generic + Aliases)
    print("\n--- Testing tool_list_pages (Generic) ---")
    
    # Int ID
    main_pages = await tool_list_pages(index, namespace=NS_MAIN, prefix="Phys")
    print(f"Main Pages (Phys*): {main_pages}")
    assert "Physics" in main_pages
    
    # String Alias (Category)
    # Note: verify_schema_tools calls tool_list_pages directly, which bypasses the specific logic in base.py where parsing happens.
    # To test base.py logic, we'd need to invoke _handle_list_pages or duplicate logic.
    # However, FaissIndex supports None. Let's test None.
    print("Testing tool_list_pages (namespace=None)...")
    all_pages = await tool_list_pages(index, namespace=None, prefix="Phys")
    print(f"All Pages (Phys*): {all_pages}")
    # Should Include "Physics" (Main) and "Category:Physics" (Cat) if it existed (it does).
    assert "Physics" in all_pages
    assert "Category:Physics" in all_pages

    # 5. Test JSON Serialization (No Text)
    print("\n--- Testing serialization (No Text) ---")
    index.save()
    import json
    with open("temp_meta.json") as f:
        meta = json.load(f)
        first_doc = list(meta["doc_map"].values())[0]
        print(f"Serialized Doc Keys: {first_doc.keys()}")
        if "text" in first_doc:
            print("FAILURE: 'text' field present in metadata!")
        else:
            print("SUCCESS: 'text' field excluded.")
            
    # Cleanup
    if Path("temp_index.bin").exists(): os.remove("temp_index.bin")
    if Path("temp_meta.json").exists(): os.remove("temp_meta.json")
    
    print("\nVERIFICATION COMPLETE")

if __name__ == "__main__":
    asyncio.run(main())

from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from tqdm import tqdm
import torch
import chromadb
import numpy as np
import gradio as gr
from PIL import Image
import io
import time

class CLIPHandler:
    def __init__(self, model, processor, device=None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.processor = processor

    def clip_embed(self, images=None, text=None):
        if not images and not text:
            raise ValueError("Specify either text or image")
        
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            if not images:
                embeddings = self.model.get_text_features(**inputs)
            elif not text:
                embeddings = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                embeddings = outputs.text_embeds if text else outputs.image_embeds
                
        return embeddings

    def clip_embed_batch(self, images):
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        
        return outputs.cpu().numpy()

    def embed_text_query(self, text):
        start_time = time.time()
        text_embedding = self.clip_embed(text=text)
        elapsed = time.time() - start_time
        print(f"Text embedding time: {elapsed:.2f} seconds")
        return text_embedding.cpu().numpy()

# Detect if CUDA is available and print GPU info
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("CUDA not available, using CPU")

def populate_chromadb(ds, limit=1000):
    # First check if the dataset loaded correctly
    if ds is None or "train" not in ds or len(ds["train"]) == 0:
        print("Error: Dataset not loaded correctly or 'train' split is empty")
        return
    
    # Check if collection is already populated
    current_count = collection.count()
    if current_count > 0:
        print(f"Collection already has {current_count} entries. Skipping population.")
        return
    
    batch_size = 500
    embeddings_list = []
    metadata_list = []
    ids_list = []
    count = 0
    
    print("Available keys:", ds["train"][0].keys()) 

    print("Adding embeddings to ChromaDB...")
    start_time = time.time()
    
    for i, example in enumerate(ds["train"]):
        if i >= limit:  # Simplified condition
            break
            
        # Check if embedding exists
        if "clip_embeddings" not in example:
            print(f"Warning: No embeddings found for item {i}, skipping")
            continue
        
        # Get embedding and convert to list for ChromaDB
        embedding = example["clip_embeddings"]
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        # Create metadata with all relevant fields
        metadata = {
            "dataset_idx": i,
            "image_id": str(example.get("id", f"img_{i}")),
        }
        
        embeddings_list.append(embedding)
        metadata_list.append(metadata)
        ids_list.append(f"embedding_{i}")
        
        count += 1
        
        # Add in batches to ChromaDB
        if count % batch_size == 0 or i >= limit-1:
            try:
                collection.add(
                    embeddings=embeddings_list,
                    metadatas=metadata_list,
                    ids=ids_list
                )
                
                print(f"Added batch of {len(embeddings_list)} embeddings to ChromaDB")
                
                embeddings_list = []
                metadata_list = []
                ids_list = []
            except Exception as e:
                print(f"Error adding batch to ChromaDB: {e}")
            
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            print(f"Progress: {count}/{min(limit, len(ds['train']))} embeddings ({rate:.2f} items/sec)")

    total_time = time.time() - start_time
    print(f"Total embeddings added to ChromaDB: {count} in {total_time:.2f} seconds")

def query_chromadb_with_text(query_text, n_results=20):
    # Get text embedding
    query_embedding = clip_model.embed_text_query(query_text)
    
    # Query ChromaDB and time it
    start_time = time.time()
    
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    query_time = time.time() - start_time
    print(f"ChromaDB query time: {query_time:.2f} seconds")
    
    return results

def get_original_data(indices):
    start_time = time.time()
    original_items = ds["train"][indices]
    retrieval_time = time.time() - start_time
    print(f"Data retrieval time: {retrieval_time:.2f} seconds for {len(indices)} items")
    
    return original_items

def search_images(query_text, num_results=5):
    if not query_text.strip():
        return [None] * num_results, [None] * num_results
    
    overall_start = time.time()
    
    results = query_chromadb_with_text(query_text, num_results)
        
    metadatas = results.get("metadatas", [[]])[0]
    
    # Extract dataset indices
    dataset_indices = [int(metadata["dataset_idx"]) for metadata in metadatas]
    
    # Get original data
    original_items = get_original_data(dataset_indices)
    
    images = original_items["image"]
    
    overall_time = time.time() - overall_start
    print(f"Total search time: {overall_time:.2f} seconds")
        
    return images

# Create Gradio interface
with gr.Blocks(title="CLIP Image Search") as demo:
    gr.Markdown("# Recherche d'image avec CLIP")
    gr.Markdown("Entrer un texte **en anglais** pour chercher dans le jeu de données FLAIR.")
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="Text de recherche", placeholder="Type something like 'urban area with buildings'")
            
            gr.Examples(
                examples=[
                    ["swimming pool in garden"],
                    ["golf course"],
                    ["train tracks"],
                    ["parking lot"],
                    ["dense pine forest"]
                ],
                inputs=[text_input]
            )
            
            search_button = gr.Button("Rechercher")

    with gr.Row():
        gallery = gr.Gallery(label="Search Results", columns=5, height=600)
    
    def search_and_display(query):      
        try:
            images = search_images(query, 10)
            return images
        except Exception as e:
            return None, f"Error: {str(e)}", f"Error occurred: {str(e)}"
    
    search_button.click(
        search_and_display,
        inputs=[text_input],
        outputs=[gallery]
    )

print("Loading dataset...")
ds = load_dataset("IGNF/FLAIR_1_osm_clip")

client = chromadb.PersistentClient(path="chroma.db")
try:
    collection = client.get_collection(name="clip_embeddings")
    print(f"Using existing ChromaDB collection with {collection.count()} entries")
except:
    collection = client.create_collection(name="clip_embeddings")
    print("Created new ChromaDB collection")

    # Populate ChromaDB on first run
    populate_chromadb(ds, limit=999999)

print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPHandler(model, processor)

# Launch the interface
if __name__ == "__main__":
    print("Starting Gradio interface...")
    # Check CUDA memory usage before launching
    if torch.cuda.is_available():
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    demo.launch(server_name="0.0.0.0", server_port=8080)

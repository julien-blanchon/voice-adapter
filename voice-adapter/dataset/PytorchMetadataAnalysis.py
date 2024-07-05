import torch
import io

# Path to a sample .pt file
sample_pt_file = "/scratch-2/final_embeddings/train_embeddings_chunk_00017.pt"

# Load the .pt file
data = torch.load(sample_pt_file)

# Check the first element to understand the structure
first_element = data[0]
print("First element structure:", first_element)

# Serialize the first element to measure its size
buffer = io.BytesIO()
torch.save(first_element, buffer)
metadata_size = buffer.tell()
print("Estimated metadata size:", metadata_size)

# Alternatively, we can inspect the size of the data itself
print("Number of elements in the file:", len(data))

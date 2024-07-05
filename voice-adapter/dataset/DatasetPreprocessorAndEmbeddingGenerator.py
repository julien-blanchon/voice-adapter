import os
import torch
import json
import logging
import tarfile
import glob
import time
import argparse
import asyncio
import aiofiles
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

_HABANA_AVAILABLE = os.getenv("HABANA", "0") == "1"

if _HABANA_AVAILABLE:
    import habana_frameworks.torch.core as htcore  # type: ignore

    # Set up HQT environment
    htcore.hpu_set_env()

# Constants
RAW_DATA_DIR = "/scratch-2/Emo-2-SNAC"
EXTRACTED_DATA_DIR = "/scratch-2/extracted"
PROCESSED_DATA_DIR = "/scratch-2/processed"
FINAL_EMBEDDINGS_DIR = "/scratch-2/final_embeddings"
BATCH_SIZE = 128
MAX_LENGTH = 256
HF_TOKEN = "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"
CHUNK_SIZE = 100000

# Ensure directories exist
for directory in [
    RAW_DATA_DIR,
    EXTRACTED_DATA_DIR,
    PROCESSED_DATA_DIR,
    FINAL_EMBEDDINGS_DIR,
]:
    os.makedirs(directory, exist_ok=True)


def extract_tar(tar_file, max_retries=3):
    split = "train" if "/train/" in tar_file else "test"
    extract_path = os.path.join(EXTRACTED_DATA_DIR, split)
    os.makedirs(extract_path, exist_ok=True)

    for attempt in range(max_retries):
        try:
            with tarfile.open(tar_file, "r") as tar:
                tar.extractall(path=extract_path)
        except Exception as e:
            logging.error(f"Error extracting {tar_file}: {str(e)}")
            if attempt < max_retries - 1:
                logging.info("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logging.error(
                    f"Failed to extract {tar_file} after {max_retries} attempts"
                )
                return tar_file


def parallel_extract_tars(tar_files):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(extract_tar, tar_files),
                total=len(tar_files),
                desc="Extracting tar files",
            )
        )

    failed_extractions = [result for result in results if result is not None]
    if failed_extractions:
        logging.error(f"Failed to extract the following files: {failed_extractions}")
    else:
        logging.info("All tar files extracted successfully")


def preprocess_sample(json_data, txt_data):
    text = json_data.get("text", "")
    transcript = json_data.get("transcript", "")
    combined_text = f"Text: {text} Transcript: {transcript}"
    snac_tokens = txt_data.split()
    return {"text": combined_text, "snac_tokens": snac_tokens}


async def save_preprocessed_chunk(data, filename):
    async with aiofiles.open(filename, "w") as f:
        await f.write(json.dumps(data))


def process_file_pair(json_file, txt_file):
    try:
        with open(json_file, "r") as jf, open(txt_file, "r") as tf:
            json_data = json.load(jf)
            txt_data = tf.read()
        return preprocess_sample(json_data, txt_data)
    except Exception as e:
        logging.error(
            f"Error processing file pair {json_file} and {txt_file}: {str(e)}"
        )
        return None


def get_chunk_filename(split, chunk_index):
    return os.path.join(
        PROCESSED_DATA_DIR, f"{split}_data_chunk_{chunk_index:05d}.json"
    )


async def preprocess_and_save_split(split, start_chunk=0):
    logging.info(
        f"Starting preprocessing for {split} split from chunk {start_chunk}..."
    )
    extract_path = os.path.join(EXTRACTED_DATA_DIR, split)

    if not os.path.exists(extract_path) or not os.listdir(extract_path):
        logging.info(f"Extracting data for {split} split...")
        os.makedirs(extract_path, exist_ok=True)
        tar_files = glob.glob(os.path.join(RAW_DATA_DIR, split, "*.tar"))
        logging.info(f"Found {len(tar_files)} tar files for {split} split")
        parallel_extract_tars(tar_files)
    else:
        logging.info(f"Extracted data for {split} already exists. Skipping extraction.")

    logging.info(f"Processing JSON and TXT files for {split} split...")
    json_files = sorted(glob.glob(os.path.join(extract_path, "*.json")))
    logging.info(f"Found {len(json_files)} JSON files to process.")

    total_chunks = (len(json_files) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_index in range(start_chunk, total_chunks):
        chunk_start = chunk_index * CHUNK_SIZE
        chunk_end = min((chunk_index + 1) * CHUNK_SIZE, len(json_files))
        chunk_files = json_files[chunk_start:chunk_end]

        preprocessed_data = []
        with ProcessPoolExecutor() as executor:
            futures = []
            for json_file in chunk_files:
                txt_file = json_file.replace(".json", ".txt")
                if os.path.exists(txt_file):
                    futures.append(
                        executor.submit(process_file_pair, json_file, txt_file)
                    )
                else:
                    logging.warning(f"Missing TXT file for JSON: {json_file}")

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Preprocessing {split} chunk {chunk_index}",
            ):
                try:
                    result = future.result()
                    if result is not None:
                        preprocessed_data.append(result)
                except Exception as e:
                    logging.error(f"Error processing file pair: {str(e)}")

        filename = get_chunk_filename(split, chunk_index)
        await save_preprocessed_chunk(preprocessed_data, filename)
        logging.info(f"Chunk {chunk_index} saved to {filename}")


def run_preprocessing(start_chunk=0):
    logging.info("Starting preprocessing phase...")
    loop = asyncio.get_event_loop()
    tasks = [
        preprocess_and_save_split(split, start_chunk) for split in ["train", "test"]
    ]
    try:
        loop.run_until_complete(asyncio.gather(*tasks))
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {str(e)}")
    logging.info("Preprocessing phase complete.")


class SNACChunkDataset(torch.utils.data.Dataset):
    def __init__(self, chunk_file, tokenizer, max_length=MAX_LENGTH):
        with open(chunk_file, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [0] * padding_length
        attention_mask = attention_mask + [0] * padding_length

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "index": idx,  # Keep track of the original index
        }


def process_chunk(chunk_file, output_file, model, tokenizer, rank=0, world_size=1):
    device = "hpu"
    logging.info(f"Processing chunk {chunk_file} on device {device}")

    try:
        dataset = SNACChunkDataset(chunk_file, tokenizer)

        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
        else:
            sampler = None

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        processed_samples = []
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Processing chunk {chunk_file} on HPU {rank}")
        ):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                indices = batch["index"]

                with torch.inference_mode():
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    sequence_hidden_states = outputs.hidden_states[-1].cpu()

                for i, idx in enumerate(indices):
                    processed_samples.append(
                        {
                            "embedding": sequence_hidden_states[i],
                            "snac_tokens": dataset.data[idx.item()]["snac_tokens"],
                            "attention_mask": attention_mask[i].cpu(),
                        }
                    )

                del outputs, sequence_hidden_states

            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                logging.exception("Exception details:")
                continue

        torch.save(processed_samples, output_file)
        logging.info(
            f"Saved {len(processed_samples)} processed samples to {output_file}"
        )
        del dataset, dataloader, sampler, processed_samples

    except Exception as e:
        logging.error(f"Error in process_chunk: {e}")
        logging.exception("Exception details:")


def run_embedding_processing(start_chunk=0):
    logging.info(f"Starting embedding processing phase from chunk {start_chunk}...")

    try:
        if _HABANA_AVAILABLE:
            import habana_frameworks.torch.distributed.hccl  # type: ignore

            world_size, rank, local_rank = (
                habana_frameworks.torch.distributed.hccl.initialize_distributed_hpu()
            )
            logging.info(
                f"Initialized distributed HPU with world_size={world_size}, rank={rank}, local_rank={local_rank}"
            )
        else:
            # raise NotImplementedError("Classical PyTorch not supported")
            world_size, rank, local_rank = 1, 0, 0
    except RuntimeError as e:
        logging.warning(f"Failed to initialize distributed HPU: {e}")
        world_size, rank, local_rank = 1, 0, 0

    if rank == -1:
        rank = 0
        world_size = 1

    logging.info(f"Using rank {rank} out of world_size {world_size}")

    if not os.path.exists(PROCESSED_DATA_DIR):
        logging.error(f"Processed data directory does not exist: {PROCESSED_DATA_DIR}")
        return

    all_files = os.listdir(PROCESSED_DATA_DIR)
    logging.info(f"Files in {PROCESSED_DATA_DIR}: {all_files}")

    # Load model and tokenizer prior to the split so as to only load them once.
    device = "hpu"
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        device_map=device,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    for split in ["train", "test"]:
        chunk_files = sorted(
            glob.glob(os.path.join(PROCESSED_DATA_DIR, f"{split}_data_chunk_*.json"))
        )
        logging.info(f"Found {len(chunk_files)} chunk files for {split} split")

        if not chunk_files:
            logging.warning(f"No preprocessed chunks found for {split} split.")
            continue

        logging.info(f"Processing {split} split...")
        for i, chunk_file in enumerate(chunk_files[start_chunk:], start=start_chunk):
            logging.info(f"Checking chunk file {i}: {chunk_file}")
            if i % world_size == rank:
                output_file = os.path.join(
                    FINAL_EMBEDDINGS_DIR, f"{split}_embeddings_chunk_{i:05d}.pt"
                )
                logging.info(f"Processing chunk file: {chunk_file}")
                try:
                    process_chunk(
                        chunk_file, output_file, model, tokenizer, rank, world_size
                    )
                    logging.info(f"Successfully processed chunk file: {chunk_file}")
                except Exception as e:
                    logging.error(f"Error processing chunk file {chunk_file}: {e}")
                    logging.exception("Exception details:")
            else:
                logging.info(f"Skipping chunk file {chunk_file} due to rank/world_size")

    logging.info("Embedding processing phase complete.")

    if os.path.exists(FINAL_EMBEDDINGS_DIR):
        final_embeddings = os.listdir(FINAL_EMBEDDINGS_DIR)
        logging.info(f"Contents of {FINAL_EMBEDDINGS_DIR}: {final_embeddings}")
    else:
        logging.error(
            f"Final embeddings directory does not exist: {FINAL_EMBEDDINGS_DIR}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate embeddings for SNAC dataset with chunking."
    )
    parser.add_argument(
        "--start_chunk",
        type=int,
        default=0,
        help="Chunk index to start processing from",
    )
    parser.add_argument(
        "--start_embedding_chunk",
        type=int,
        default=3,
        help="Embedding chunk index to start processing from",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Script started")

    try:
        logging.info("Attempting to log in to Hugging Face")
        login(token=HF_TOKEN)
        logging.info("Successfully logged in to Hugging Face")

        # logging.info("Starting preprocessing")
        # run_preprocessing(args.start_chunk)
        # logging.info("Preprocessing completed")

        logging.info("Starting embedding processing")
        run_embedding_processing(args.start_embedding_chunk)
        logging.info("Embedding processing completed")
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        logging.exception("Exception details:")

    logging.info("Script finished")


if __name__ == "__main__":
    main()

logging.info(
    f"Full processing complete. Final embeddings saved in: {FINAL_EMBEDDINGS_DIR}"
)

import os
import torch
from transformers import AutoTokenizer, AutoModel
from snac import SNAC
import soundfile as sf
from tqdm import tqdm


_HABANA_AVAILABLE = os.getenv("HABANA", "0") == "1"

if _HABANA_AVAILABLE:
    from habana_frameworks.torch.hpex.experimental.transformer_engine import recipe  # type: ignore

    # Create an FP8 recipe
    fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID)
    fp8_context = lambda: fp8_context()
else:
    from torch.cuda.amp import autocast

    fp8_recipe = None
    fp8_context = lambda: autocast(enabled=True)


# Import the model and config from the training script
from SNACGeneratorModel import SNACGenerator, SNACConfig

# Constants
MAX_LENGTH = 512
DEVICE = torch.device("hpu")
SNAC_VOCAB_SIZE = 4096

# Load LLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
llm = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B").to(DEVICE)

# Load the trained SNAC Generator
config = SNACConfig()
snac_generator = SNACGenerator(config).to(DEVICE)
snac_generator.load_state_dict(torch.load("best_snac_generator.pth"))
snac_generator.eval()

# Initialize SNAC Decoder
snac_decoder = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(DEVICE)


def generate_audio(text, transcript, max_length=1000, x_tokens=10, y_tokens=5):
    # Prepare input for LLM
    input_text = f"Description: {text} Transcript: {transcript}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # Generate embeddings with LLM
    with torch.no_grad(), fp8_context():
        llm_output = llm(input_ids, attention_mask=attention_mask)

    embeddings = llm_output.last_hidden_state

    # Generate SNAC tokens
    generated_snac = []
    current_input = embeddings

    with torch.no_grad():
        for i in tqdm(range(0, max_length, x_tokens), desc="Generating SNAC tokens"):
            position_ids = torch.arange(
                current_input.size(1), dtype=torch.long, device=DEVICE
            )
            position_ids = position_ids.unsqueeze(0).expand(current_input.size(0), -1)

            with fp8_context():
                snac_logits = snac_generator(
                    current_input,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

            predicted_snac = snac_logits[:, -x_tokens:].argmax(dim=-1).squeeze(0)

            # Check for stop token
            stop_index = (predicted_snac == 0).nonzero(as_tuple=True)[0]
            if stop_index.numel() > 0:
                generated_snac.extend(predicted_snac[: stop_index[0]].tolist())
                break
            else:
                generated_snac.extend(predicted_snac.tolist())

            # Update current_input for next iteration
            snac_embeddings = snac_generator.snac_proj.weight[predicted_snac]
            current_input = torch.cat([current_input, snac_embeddings], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.size(0), x_tokens), device=DEVICE),
                ],
                dim=1,
            )

            # Generate new LLM tokens every y_tokens
            if len(generated_snac) % y_tokens == 0:
                new_llm_input = tokenizer.decode(input_ids[0]) + tokenizer.decode(
                    input_ids[0][-y_tokens:]
                )
                new_llm_inputs = tokenizer(
                    new_llm_input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding="max_length",
                )
                new_input_ids = new_llm_inputs["input_ids"].to(DEVICE)
                new_attention_mask = new_llm_inputs["attention_mask"].to(DEVICE)

                with fp8_context():
                    new_llm_output = llm(
                        new_input_ids, attention_mask=new_attention_mask
                    )

                new_llm_embeddings = new_llm_output.last_hidden_state
                current_input = torch.cat(
                    [current_input, new_llm_embeddings[:, -y_tokens:, :]], dim=1
                )
                attention_mask = torch.cat(
                    [attention_mask, new_attention_mask[:, -y_tokens:]], dim=1
                )

    # Decode SNAC tokens to audio
    with torch.inference_mode(), fp8_context():
        audio = snac_decoder.decode(generated_snac)

    return audio, generated_snac


def save_audio(audio, filename):
    sf.write(filename, audio.cpu().numpy(), 24000)


# Example usage
def main():
    text = "A speaker conveys deep sadness and disappointment."
    transcript = "Our record played outside Paola."

    audio, snac_tokens = generate_audio(text, transcript, x_tokens=10, y_tokens=5)
    save_audio(audio, "generated_audio.wav")

    print(f"Generated SNAC tokens: {snac_tokens}")
    print("Audio saved as 'generated_audio.wav'")


if __name__ == "__main__":
    main()

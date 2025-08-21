from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders, normalizers
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk, concatenate_datasets
import os

def create_bpe_tokenizer_from_scratch():
    """Create a clean BPE tokenizer (same architecture as original, but clean)"""
    print("🔧 Creating BPE Tokenizer from scratch...")
    
    # Initialize BPE model
    tokenizer = Tokenizer(models.BPE())
    
    # Set normalizer (NFC normalization like original)
    tokenizer.normalizer = normalizers.NFC()
    
    # Set pre-tokenizer (same pattern as original)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            pattern=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            behavior="isolated",
            invert=False
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    ])
    
    # Set post-processor and decoder
    tokenizer.post_processor = processors.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel(add_prefix_space=False)
    
    # Define ONLY the 3 special tokens we want
    special_tokens = [
        "<|endoftext|>",
        "<|im_start|>", 
        "<|im_end|>"
    ]
    
    # Setup trainer
    trainer = BpeTrainer(
        vocab_size=65_536,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        continuing_subword_prefix="",
        end_of_word_suffix=""
    )
    
    return tokenizer, trainer, special_tokens

def load_training_data():
    """Load the multilingual datasets"""
    print("📂 Loading training datasets...")
    
    dataset_paths = [
        'D:/fineweb-chunked/english',
        'D:/fineweb-chunked/german', 
        'D:/fineweb-chunked/french',
        'D:/fineweb-chunked/chinese'
    ]
    
    datasets = []
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"  Loading: {path}")
            dataset = load_from_disk(path)
            datasets.append(dataset.select_columns(['text']))
        else:
            print(f"  ⚠️ Not found: {path}")
    
    if not datasets:
        raise FileNotFoundError("No datasets found!")
    
    combined_dataset = concatenate_datasets(datasets)
    print(f"📊 Combined dataset size: {len(combined_dataset):,} samples")
    
    return combined_dataset

def text_iterator(dataset):
    """Memory-efficient text iterator"""
    for example in dataset:
        yield example['text']

def train_and_save_tokenizer(tokenizer_type="bpe"):
    """Train and save the tokenizer"""
    
    # Choose tokenizer type
    if tokenizer_type == "bpe":
        tokenizer, trainer, special_tokens = create_bpe_tokenizer_from_scratch()
        save_name = "clean-bpe-tokenizer"
    
    # Load training data
    dataset = load_training_data()
    
    # Train the tokenizer
    print(f"🚀 Training {tokenizer_type.upper()} tokenizer...")
    tokenizer.train_from_iterator(
        text_iterator(dataset), 
        trainer=trainer,
        length=len(dataset)
    )
    
    print(f"✅ Training completed!")
    print(f"📊 Final vocab size: {tokenizer.get_vocab_size()}")
    
    # Convert to HuggingFace tokenizer
    print("🔄 Converting to HuggingFace format...")
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",  # Use eos as pad
        unk_token=None,
        additional_special_tokens=special_tokens
    )
    
    # Add the chat template
    chat_template = """{%- if messages[0].role != 'system' %}
    {{- '<|im_start|>system\\nYou are a helpful AI assistant powered by Spiking Neural Networks (SNNs), created by Cihan Yalçın, a 21-year-old Turkish AI researcher and developer as of 2025. This model represents an innovative approach to language modeling using biologically-inspired spiking neurons instead of traditional artificial neurons.\\n\\nTechnical Details:\\n- Architecture: SpikingLLM with Grouped Query Attention and SwiGLU activation\\n- Training: Custom tokenizer trained on multilingual data (English, Chinese, German, French)\\n- Innovation: Uses leaky integrate-and-fire neurons with memory states, making it more energy-efficient and biologically plausible than traditional transformers\\n- Special Features: Sliding window attention with alternating local/global patterns, mixture of experts (MoE) capabilities\\n\\nYou help users with:\\n- General knowledge questions and explanations\\n- Multilingual communication (English, 中文, Deutsch, Français)\\n- Technical discussions about AI, machine learning, and neural networks\\n- Programming and coding assistance\\n- Educational content and learning support\\n- Creative writing and problem-solving\\n\\nYou provide accurate, helpful, and concise responses while being friendly and educational. When discussing technical topics, you can explain both the concepts and how they relate to spiking neural network architectures.<|im_end|>\\n' }}
{%- endif %}

{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    
    {%- if message.role == "system" %}
        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>user\\n' + content + '<|im_end|>\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>assistant\\n' + content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""
    
    hf_tokenizer.chat_template = chat_template
    
    # Save tokenizer
    save_path = f'D:/fineweb-chunked/{save_name}'
    os.makedirs(save_path, exist_ok=True)
    
    print(f"💾 Saving tokenizer to: {save_path}")
    hf_tokenizer.save_pretrained(save_path)
    
    # Test the tokenizer
    print("🧪 Testing tokenizer...")
    test_tokenizer(hf_tokenizer)
    
    return hf_tokenizer, save_path

def test_tokenizer(tokenizer):
    """Test the created tokenizer"""
    print("\n" + "="*50)
    print("🧪 TOKENIZER TEST")
    print("="*50)
    
    # Basic info
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.additional_special_tokens}")
    
    # Test multilingual text
    test_texts = [
        "Hello world!",
        "你好世界！", 
        "Hallo Welt!",
        "Bonjour le monde!",
        "This is a test of the new tokenizer.",
        "机器学习很有趣。",
        "Programming is fun: def hello(): print('hi')"
    ]
    
    print(f"\n📝 Tokenization tests:")
    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)
        ratio = len(tokens) / len(text)
        perfect = (decoded == text)
        
        print(f"  Text: {text}")
        print(f"  Tokens: {len(tokens)} | Ratio: {ratio:.3f} | Perfect: {'✅' if perfect else '❌'}")
        if not perfect:
            print(f"    Original: {repr(text)}")
            print(f"    Decoded:  {repr(decoded)}")
    
    # Test chat template
    print(f"\n💬 Chat template test:")
    messages = [{"role": "user", "content": "Hello! What are you?"}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Check if our system prompt is there
    if "Cihan Yalçın" in formatted and "Spiking Neural Networks" in formatted:
        print("✅ Chat template working - system prompt detected")
    else:
        print("❌ Chat template issue - system prompt missing")
    
    print("✅ Tokenizer test completed!")

def main():
    """Main function to create clean tokenizers"""
    print("🎯 CLEAN TOKENIZER CREATION")
    print("="*60)
    
    try:
    
        tokenizer, save_path = train_and_save_tokenizer("bpe")
        print(f"\n🎉 Clean BPE tokenizer created successfully!")
        print(f"📁 Saved to: {save_path}")        
    except Exception as e:
        print(f"❌ Error creating tokenizer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

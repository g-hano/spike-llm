from transformers import AutoTokenizer
import json
import os
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_tokenizer():
    """Load tokenizer"""
    tokenizer_path = 'D:/fineweb-chunked/clean-bpe-tokenizer'
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")
    
    print(f"🔄 Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("✅ Tokenizer loaded successfully!")
    return tokenizer

def analyze_basic_properties(tokenizer):
    """Comprehensive basic analysis"""
    print("\n" + "="*80)
    print("📊 BASIC TOKENIZER PROPERTIES")
    print("="*80)
    
    # Basic info
    vocab = tokenizer.get_vocab()
    
    basic_info = {
        "Vocabulary Size": len(vocab),
        "Model Type": type(tokenizer).__name__,
        "Model Max Length": getattr(tokenizer, 'model_max_length', 'Unknown'),
        "Padding Side": getattr(tokenizer, 'padding_side', 'Unknown'),
        "Truncation Side": getattr(tokenizer, 'truncation_side', 'Unknown'),
    }
    
    for key, value in basic_info.items():
        print(f"{key:20}: {value}")
    
    # Special tokens analysis
    print(f"\n🔥 SPECIAL TOKENS ANALYSIS:")
    special_tokens_info = {
        "EOS Token": tokenizer.eos_token,
        "EOS Token ID": tokenizer.eos_token_id,
        "PAD Token": tokenizer.pad_token,
        "PAD Token ID": getattr(tokenizer, 'pad_token_id', None),
        "UNK Token": tokenizer.unk_token,
        "UNK Token ID": getattr(tokenizer, 'unk_token_id', None),
        "BOS Token": getattr(tokenizer, 'bos_token', None),
        "BOS Token ID": getattr(tokenizer, 'bos_token_id', None),
    }
    
    for key, value in special_tokens_info.items():
        print(f"{key:20}: {value}")
    
    print(f"\n📝 ADDITIONAL SPECIAL TOKENS ({len(tokenizer.additional_special_tokens)}):")
    for i, token in enumerate(tokenizer.additional_special_tokens):
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {i+1:2d}. {token:15} (ID: {token_id})")
    
    return vocab, basic_info

def analyze_vocabulary_distribution(tokenizer, vocab):
    """Detailed vocabulary analysis"""
    print("\n" + "="*80)
    print("📚 VOCABULARY DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Analyze token types
    token_types = {
        'single_char': 0,
        'multi_char': 0,
        'numeric': 0,
        'punctuation': 0,
        'special': 0,
        'whitespace': 0,
        'mixed': 0
    }
    
    token_lengths = []
    decoded_tokens = []
    
    print("🔍 Sampling and analyzing vocabulary...")
    
    # Sample tokens for analysis (analyze every 100th token to get representative sample)
    sample_ids = list(range(0, len(vocab), max(1, len(vocab) // 1000)))
    
    for token_id in sample_ids:
        try:
            # Decode individual token
            decoded = tokenizer.decode([token_id])
            decoded_tokens.append(decoded)
            token_lengths.append(len(decoded))
            
            # Categorize token type
            if decoded in tokenizer.additional_special_tokens or decoded in [tokenizer.eos_token, tokenizer.pad_token, tokenizer.unk_token]:
                token_types['special'] += 1
            elif decoded.isspace():
                token_types['whitespace'] += 1
            elif decoded.isdigit():
                token_types['numeric'] += 1
            elif decoded.isalpha() and len(decoded) == 1:
                token_types['single_char'] += 1
            elif decoded.isalpha() and len(decoded) > 1:
                token_types['multi_char'] += 1
            elif all(c in '.,!?;:()[]{}"\'-' for c in decoded):
                token_types['punctuation'] += 1
            else:
                token_types['mixed'] += 1
                
        except Exception as e:
            print(f"⚠️ Error decoding token {token_id}: {e}")
            continue
    
    # Print token type distribution
    print(f"\n📊 TOKEN TYPE DISTRIBUTION (sample of {len(sample_ids)} tokens):")
    total_sampled = sum(token_types.values())
    for token_type, count in token_types.items():
        percentage = (count / total_sampled * 100) if total_sampled > 0 else 0
        print(f"  {token_type:12}: {count:5d} ({percentage:5.1f}%)")
    
    # Token length analysis
    if token_lengths:
        print(f"\n📏 TOKEN LENGTH STATISTICS:")
        print(f"  Average length: {np.mean(token_lengths):.2f} characters")
        print(f"  Median length:  {np.median(token_lengths):.2f} characters")
        print(f"  Min length:     {min(token_lengths)} characters")
        print(f"  Max length:     {max(token_lengths)} characters")
        print(f"  Std deviation:  {np.std(token_lengths):.2f}")
    
    return token_types, token_lengths, decoded_tokens

def test_multilingual_capabilities(tokenizer):
    """Comprehensive multilingual testing"""
    print("\n" + "="*80)
    print("🌍 MULTILINGUAL CAPABILITIES TEST")
    print("="*80)
    
    # Comprehensive test texts for each language
    test_cases = {
        "English": [
            "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "I can't believe it's already 2024!",
            "Programming languages include Python, JavaScript, and C++."
        ],
        "Chinese": [
            "你好，今天过得怎么样？",
            "机器学习是人工智能的一个分支。",
            "北京是中国的首都。",
            "我正在学习深度学习和神经网络。",
            "春天来了，樱花开了。"
        ],
        "German": [
            "Hallo, wie geht es dir heute?",
            "Maschinelles Lernen ist ein Teilbereich der künstlichen Intelligenz.",
            "Berlin ist die Hauptstadt von Deutschland.",
            "Ich lerne gerade über neuronale Netzwerke.",
            "Der Frühling kommt und die Blumen blühen."
        ],
        "French": [
            "Bonjour, comment allez-vous aujourd'hui?",
            "L'apprentissage automatique est une branche de l'intelligence artificielle.",
            "Paris est la capitale de la France.",
            "J'apprends actuellement les réseaux de neurones.",
            "Le printemps arrive et les fleurs fleurissent."
        ]
    }
    
    results = []
    
    for language, texts in test_cases.items():
        print(f"\n🔤 {language.upper()} TESTS:")
        print("-" * 50)
        
        lang_results = {
            'language': language,
            'total_chars': 0,
            'total_tokens': 0,
            'perfect_decodes': 0,
            'total_tests': len(texts)
        }
        
        for i, text in enumerate(texts, 1):
            # Tokenize
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            
            # Calculate metrics
            compression_ratio = len(tokens) / len(text)
            perfect_decode = decoded.strip() == text.strip()
            
            # Accumulate stats
            lang_results['total_chars'] += len(text)
            lang_results['total_tokens'] += len(tokens)
            if perfect_decode:
                lang_results['perfect_decodes'] += 1
            
            print(f"  Test {i:2d}: {text}")
            print(f"    Tokens: {len(tokens):3d} | Ratio: {compression_ratio:.3f} | Perfect: {'✅' if perfect_decode else '❌'}")
            
            if not perfect_decode:
                print(f"    Original: {repr(text)}")
                print(f"    Decoded:  {repr(decoded)}")
            
            # Show some tokens for first test
            if i == 1:
                token_list = tokenizer.tokenize(text)
                print(f"    Token breakdown: {token_list[:10]}{'...' if len(token_list) > 10 else ''}")
        
        # Calculate language-specific metrics
        lang_results['avg_compression'] = lang_results['total_tokens'] / lang_results['total_chars']
        lang_results['perfect_decode_rate'] = lang_results['perfect_decodes'] / lang_results['total_tests']
        
        results.append(lang_results)
    
    # Summary
    print(f"\n📈 MULTILINGUAL PERFORMANCE SUMMARY:")
    print("-" * 60)
    print(f"{'Language':<10} {'Avg Ratio':<12} {'Perfect Rate':<15} {'Total Tokens'}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['language']:<10} {result['avg_compression']:<12.4f} {result['perfect_decode_rate']*100:<13.1f}% {result['total_tokens']:>8}")
    
    return results

def test_special_use_cases(tokenizer):
    """Test tokenizer on special cases and edge scenarios"""
    print("\n" + "="*80)
    print("🔍 SPECIAL USE CASES & EDGE SCENARIOS")
    print("="*80)
    
    test_scenarios = {
        "Contractions": [
            "I'm can't won't shouldn't don't",
            "We're they're you're it's that's"
        ],
        "URLs & Emails": [
            "Visit https://www.example.com for more info",
            "Contact us at hello@company.com or support@test.org"
        ],
        "Numbers & Math": [
            "The numbers are 123, 456.789, and 1,000,000",
            "Calculate 3.14159 × 2.71828 = 8.539"
        ],
        "Programming": [
            "def hello_world(): print('Hello, World!')",
            "const data = {name: 'test', value: 42};"
        ],
        "Mixed Scripts": [
            "Hello 你好 Guten Tag Bonjour!",
            "AI人工智能 ML机器学习 programming编程"
        ],
        "Special Characters": [
            "Symbols: @#$%^&*()_+-=[]{}|;':\",./<>?",
            "Unicode: café résumé naïve 北京 東京"
        ],
        "Instruction Format": [
            "<|im_start|>user\nWhat is machine learning?<|im_end|>",
            "<|im_start|>assistant\nMachine learning is...<|im_end|>"
        ],
        "Long Repetitive": [
            "ha " * 20,  # Test repetitive patterns
            "test " * 15 + "final"
        ]
    }
    
    for category, test_texts in test_scenarios.items():
        print(f"\n🧪 {category.upper()}:")
        print("-" * 40)
        
        for text in test_texts:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            token_breakdown = tokenizer.tokenize(text)
            
            perfect = decoded.strip() == text.strip()
            compression = len(tokens) / len(text) if len(text) > 0 else 0
            
            print(f"  Text: {text}")
            print(f"  Tokens ({len(tokens)}): {token_breakdown}")
            print(f"  Compression: {compression:.3f} | Perfect: {'✅' if perfect else '❌'}")
            
            if not perfect:
                print(f"  ⚠️ Decode issue:")
                print(f"    Original: {repr(text)}")
                print(f"    Decoded:  {repr(decoded)}")
            print()

def analyze_token_efficiency(tokenizer):
    """Analyze tokenization efficiency patterns"""
    print("\n" + "="*80)
    print("⚡ TOKENIZATION EFFICIENCY ANALYSIS")
    print("="*80)
    
    # Test various word types and patterns
    efficiency_tests = {
        "Common English Words": ["the", "and", "that", "have", "for", "not", "with", "you", "this", "but"],
        "Long English Words": ["international", "understanding", "responsibility", "administration", "characteristic"],
        "Chinese Characters": ["的", "一", "是", "在", "有", "人", "他", "这", "中", "大"],
        "Chinese Words": ["机器学习", "人工智能", "深度学习", "神经网络", "自然语言"],
        "German Compounds": ["Maschinelles", "Kunstintelligenz", "Sprachverarbeitung", "Datenverarbeitung"],
        "French Words": ["apprentissage", "intelligence", "artificielle", "traitement", "langue"],
        "Technical Terms": ["tokenization", "transformer", "attention", "embedding", "gradient"],
        "Numbers": ["123", "1.5", "1,000", "3.14159", "2024"],
    }
    
    results = {}
    
    for category, words in efficiency_tests.items():
        print(f"\n📊 {category}:")
        
        category_stats = {
            'total_chars': 0,
            'total_tokens': 0,
            'word_count': len(words),
            'ratios': []
        }
        
        for word in words:
            tokens = tokenizer.encode(word, add_special_tokens=False)  # Don't add special tokens for individual words
            token_breakdown = tokenizer.tokenize(word)
            
            ratio = len(tokens) / len(word) if len(word) > 0 else 0
            category_stats['total_chars'] += len(word)
            category_stats['total_tokens'] += len(tokens)
            category_stats['ratios'].append(ratio)
            
            print(f"  {word:15} → {len(tokens)} tokens ({ratio:.3f}) | {token_breakdown}")
        
        avg_ratio = category_stats['total_tokens'] / category_stats['total_chars']
        print(f"  📈 Category average: {avg_ratio:.4f} tokens/char")
        
        results[category] = category_stats
    
    # Overall efficiency summary
    print(f"\n📊 EFFICIENCY SUMMARY:")
    print("-" * 50)
    for category, stats in results.items():
        avg_ratio = stats['total_tokens'] / stats['total_chars']
        print(f"{category:20}: {avg_ratio:.4f} tokens/char")
    
    return results

def save_analysis_results(tokenizer, all_results):
    """Save comprehensive analysis results"""
    print("\n" + "="*80)
    print("💾 SAVING ANALYSIS RESULTS")
    print("="*80)
    
    output_dir = Path("D:/fineweb-chunked/tokenizer3_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Save tokenizer info
    tokenizer_info = {
        "vocab_size": len(tokenizer.get_vocab()),
        "model_type": type(tokenizer).__name__,
        "special_tokens": {
            "eos_token": tokenizer.eos_token,
            "pad_token": tokenizer.pad_token,
            "unk_token": tokenizer.unk_token,
            "additional_special_tokens": tokenizer.additional_special_tokens
        }
    }
    
    with open(output_dir / "tokenizer_info.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_info, f, indent=2, ensure_ascii=False)
    
    # Save results to JSON
    with open(output_dir / "analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✅ Analysis results saved to: {output_dir}")
    print(f"📁 Files created:")
    print(f"  - tokenizer_info.json")
    print(f"  - analysis_results.json")

def main():
    """Main analysis function"""
    print("🔬 COMPREHENSIVE TOKENIZER ANALYSIS")
    print("="*80)
    
    try:
        # Load tokenizer
        tokenizer = load_tokenizer()
        
        # Run all analyses
        print("🔄 Running comprehensive analysis...")
        
        all_results = {}
        
        # Basic properties
        vocab, basic_info = analyze_basic_properties(tokenizer)
        all_results['basic_info'] = basic_info
        
        # Vocabulary analysis
        token_types, token_lengths, decoded_samples = analyze_vocabulary_distribution(tokenizer, vocab)
        all_results['vocabulary_analysis'] = {
            'token_types': token_types,
            'token_length_stats': {
                'mean': float(np.mean(token_lengths)) if token_lengths else 0,
                'median': float(np.median(token_lengths)) if token_lengths else 0,
                'std': float(np.std(token_lengths)) if token_lengths else 0,
                'min': int(min(token_lengths)) if token_lengths else 0,
                'max': int(max(token_lengths)) if token_lengths else 0
            }
        }
        
        # Multilingual testing
        multilingual_results = test_multilingual_capabilities(tokenizer)
        all_results['multilingual_performance'] = multilingual_results
        
        # Special use cases
        test_special_use_cases(tokenizer)
        
        # Efficiency analysis
        efficiency_results = analyze_token_efficiency(tokenizer)
        all_results['efficiency_analysis'] = efficiency_results
        
        # Save results
        save_analysis_results(tokenizer, all_results)
        
        # Final summary
        print("\n" + "="*80)
        print("🎯 FINAL ASSESSMENT SUMMARY")
        print("="*80)
        
        print(f"✅ Vocabulary Size: {len(vocab):,} tokens")
        print(f"✅ Special Tokens: {len(tokenizer.additional_special_tokens)} additional")
        
        # Calculate overall multilingual performance
        total_perfect = sum(r['perfect_decodes'] for r in multilingual_results)
        total_tests = sum(r['total_tests'] for r in multilingual_results)
        overall_accuracy = total_perfect / total_tests * 100
        
        avg_compression = np.mean([r['avg_compression'] for r in multilingual_results])
        
        print(f"✅ Overall Accuracy: {overall_accuracy:.1f}% perfect decodes")
        print(f"✅ Average Compression: {avg_compression:.4f} tokens/character")
        
        print(f"\n🏆 TOKENIZER-3 ANALYSIS COMPLETE!")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

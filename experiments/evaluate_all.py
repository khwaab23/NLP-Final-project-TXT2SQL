"""
Evaluate all models and compare results
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import json
from typing import Dict, Any

from src.models import PhiModel, LlamaModel, MistralModel
from src.models import GPT4Model, GeminiModel, ClaudeModel
from src.utils.data_loader import SpiderDataset
from src.evaluation import ModelEvaluator


def load_config(config_path: str = "./config/api_keys.yaml") -> Dict[str, Any]:
    """Load API keys configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_slm_models(
    dataset,
    model_configs: Dict[str, Any],
    num_samples: int = None
) -> Dict[str, Dict]:
    """Evaluate Small Language Models"""
    results = {}
    
    # Phi-2 with LoRA
    print("\n" + "="*50)
    print("Evaluating Phi-2 + LoRA")
    print("="*50)
    phi_lora = PhiModel(model_configs['phi-2'], use_lora=True)
    phi_lora.load_adapter("./results/lora/phi-2")
    evaluator = ModelEvaluator(phi_lora, dataset)
    results['Phi-2-LoRA'] = evaluator.evaluate(num_samples=num_samples)
    
    # Phi-2 with DoRA
    print("\n" + "="*50)
    print("Evaluating Phi-2 + DoRA")
    print("="*50)
    phi_dora = PhiModel(model_configs['phi-2'], use_dora=True)
    phi_dora.load_adapter("./results/dora/phi-2")
    evaluator = ModelEvaluator(phi_dora, dataset)
    results['Phi-2-DoRA'] = evaluator.evaluate(num_samples=num_samples)
    
    # Llama-7B with LoRA
    print("\n" + "="*50)
    print("Evaluating Llama-7B + LoRA")
    print("="*50)
    llama_lora = LlamaModel(model_configs['llama-7b'], use_lora=True)
    llama_lora.load_adapter("./results/lora/llama-7b")
    evaluator = ModelEvaluator(llama_lora, dataset)
    results['Llama-7B-LoRA'] = evaluator.evaluate(num_samples=num_samples)
    
    # Mistral-7B with GRPO
    print("\n" + "="*50)
    print("Evaluating Mistral-7B + GRPO")
    print("="*50)
    mistral_grpo = MistralModel(model_configs['mistral-7b'])
    mistral_grpo.load_adapter("./results/grpo/mistral-7b")
    evaluator = ModelEvaluator(mistral_grpo, dataset)
    results['Mistral-7B-GRPO'] = evaluator.evaluate(num_samples=num_samples)
    
    return results


def evaluate_generalist_models(
    dataset,
    api_keys: Dict[str, Any],
    model_configs: Dict[str, Any],
    num_samples: int = None
) -> Dict[str, Dict]:
    """Evaluate Generalist Models (GPT-4, Gemini, Claude)"""
    results = {}
    
    # GPT-4
    print("\n" + "="*50)
    print("Evaluating GPT-4")
    print("="*50)
    gpt4 = GPT4Model(model_configs['gpt-4'], api_keys['openai']['api_key'])
    evaluator = ModelEvaluator(gpt4, dataset)
    results['GPT-4'] = evaluator.evaluate(num_samples=num_samples)
    
    # Gemini Pro
    print("\n" + "="*50)
    print("Evaluating Gemini Pro")
    print("="*50)
    gemini = GeminiModel(model_configs['gemini'], api_keys['google']['api_key'])
    evaluator = ModelEvaluator(gemini, dataset)
    results['Gemini-Pro'] = evaluator.evaluate(num_samples=num_samples)
    
    # Claude 3
    print("\n" + "="*50)
    print("Evaluating Claude 3")
    print("="*50)
    claude = ClaudeModel(model_configs['claude'], api_keys['anthropic']['api_key'])
    evaluator = ModelEvaluator(claude, dataset)
    results['Claude-3'] = evaluator.evaluate(num_samples=num_samples)
    
    return results


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table from results"""
    comparison_data = []
    
    for model_name, model_results in results.items():
        row = {
            'Model': model_name,
            'Exact Match (%)': model_results.get('exact_match_avg', 0) * 100,
            'Execution Acc (%)': model_results.get('execution_accuracy_avg', 0) * 100,
            'Component Match (%)': model_results.get('component_match_avg', 0) * 100,
            'Valid SQL (%)': model_results.get('valid_sql_avg', 0) * 100,
            'Avg Time (ms)': model_results.get('avg_inference_time', 0) * 1000,
        }
        
        # Add cost if available
        if 'model_stats' in model_results:
            row['Cost/1K queries ($)'] = model_results['model_stats'].get('total_cost', 0) * 1000
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df.sort_values('Execution Acc (%)', ascending=False)


def save_results(results: Dict, comparison_df: pd.DataFrame, output_dir: str = "./results"):
    """Save evaluation results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_path / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save comparison table
    comparison_df.to_csv(output_path / "comparison_table.csv", index=False)
    
    # Save formatted markdown table
    with open(output_path / "comparison_table.md", 'w') as f:
        f.write("# Text-to-SQL Model Comparison\n\n")
        f.write(comparison_df.to_markdown(index=False))
    
    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all Text-to-SQL models")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/spider",
        help="Path to Spider dataset"
    )
    parser.add_argument(
        "--api_keys_config",
        type=str,
        default="./config/api_keys.yaml",
        help="Path to API keys configuration"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./config/model_config.yaml",
        help="Path to model configuration"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (use smaller number for quick testing)"
    )
    parser.add_argument(
        "--skip_slm",
        action="store_true",
        help="Skip SLM evaluation"
    )
    parser.add_argument(
        "--skip_generalist",
        action="store_true",
        help="Skip generalist model evaluation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/evaluation",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("Text-to-SQL Model Evaluation")
    print("="*50)
    
    # Load configurations
    api_keys = load_config(args.api_keys_config)
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Load test dataset
    print(f"\nLoading test dataset from {args.data_path}...")
    test_dataset = SpiderDataset(
        args.data_path,
        split="dev",  # Use dev set for evaluation
        tokenizer=None,  # Will handle tokenization in each model
        include_schema=True
    )
    
    if args.num_samples:
        test_dataset.examples = test_dataset.examples[:args.num_samples]
    
    print(f"Test samples: {len(test_dataset)}")
    
    all_results = {}
    
    # Evaluate SLMs
    if not args.skip_slm:
        print("\n" + "="*50)
        print("EVALUATING SMALL LANGUAGE MODELS")
        print("="*50)
        slm_results = evaluate_slm_models(
            test_dataset,
            model_config['models']['slm'],
            args.num_samples
        )
        all_results.update(slm_results)
    
    # Evaluate Generalist Models
    if not args.skip_generalist:
        print("\n" + "="*50)
        print("EVALUATING GENERALIST MODELS")
        print("="*50)
        generalist_results = evaluate_generalist_models(
            test_dataset,
            api_keys,
            model_config['models']['generalist'],
            args.num_samples
        )
        all_results.update(generalist_results)
    
    # Create comparison table
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    comparison_df = create_comparison_table(all_results)
    print("\n", comparison_df.to_string(index=False))
    
    # Save results
    save_results(all_results, comparison_df, args.output_dir)
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()

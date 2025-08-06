"""
Local CLI version for batch processing without Streamlit.

Usage:
    python main_local.py --folder /path/to/images --metrics clip_score entropy_fairness --api-key YOUR_KEY --model-provider openai

Examples:
    # Process with CLIP score only (no API key needed)
    python main_local.py --folder ./experiment_images --metrics clip_score --gpu 0

    # Process with all metrics using OpenAI
    python main_local.py --folder ./experiment_images --metrics clip_score entropy_fairness kl_fairness --api-key sk-xxx --model-provider openai --model gpt-4o

    # Process with Gemini
    python main_local.py --folder ./experiment_images --metrics entropy_fairness --api-key YOUR_GEMINI_KEY --model-provider gemini --model gemini-2.0-flash

    # Save results to custom location
    python main_local.py --folder ./experiment_images --metrics clip_score --output ./my_results.json
"""

import os
import json
import argparse
import sys
from datetime import datetime
import warnings

from main import ImageEvaluator, get_available_metrics, get_gpu_info

warnings.filterwarnings("ignore")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process images for fairness and utility metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        required=True,
        help="Path to folder containing images (supports subdirectories)",
    )

    available_metric_keys = [m["key"] for m in get_available_metrics()]
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        choices=available_metric_keys + ["all"],
        default=["clip_score"],
        help=f"Metrics to evaluate. Available: {', '.join(available_metric_keys)}. Use 'all' for all available metrics.",
    )

    parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        help="API key for demographic analysis (required for fairness metrics)",
    )

    parser.add_argument(
        "--model-provider",
        "-p",
        choices=["openai", "gemini"],
        default="openai",
        help="API provider for demographic analysis",
    )

    parser.add_argument(
        "--model",
        "-M",
        type=str,
        help="Model name to use (e.g., gpt-4o, gemini-2.0-flash)",
    )

    parser.add_argument(
        "--gpu", "-g", type=int, help="GPU index to use (default: auto-select best GPU)"
    )

    parser.add_argument(
        "--cpu-only", action="store_true", help="Force CPU-only processing"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output JSON file path (default: auto-generated based on folder name)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--env-api-key",
        action="store_true",
        help="Use API key from environment variables (OPENAI_API_KEY, GEMINI_API_KEY)",
    )

    return parser.parse_args()


def get_env_api_key(provider):
    """Get API key from environment variables"""
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    elif provider == "gemini":
        return os.getenv("GEMINI_API_KEY", "")
    return ""


def select_best_gpu():
    """Automatically select the GPU with most free memory"""
    try:
        gpu_info = get_gpu_info()
        if not gpu_info:
            return None

        best_gpu = max(gpu_info, key=lambda gpu: gpu["memory_free"])
        return best_gpu["index"]
    except Exception:
        return None


def setup_model_defaults(provider):
    """Set default model names based on provider"""
    defaults = {"openai": "gpt-4o-2024-05-13", "gemini": "gemini-2.0-flash"}
    return defaults.get(provider, defaults["openai"])


def validate_args(args):
    """Validate command line arguments"""
    errors = []

    if not os.path.exists(args.folder):
        errors.append(f"Folder does not exist: {args.folder}")
    elif not os.path.isdir(args.folder):
        errors.append(f"Path is not a directory: {args.folder}")

    if "all" in args.metrics:
        available_metrics = get_available_metrics()
        args.metrics = [m["key"] for m in available_metrics]

    fairness_metrics = {"entropy_fairness", "kl_fairness"}
    has_fairness = any(metric in fairness_metrics for metric in args.metrics)

    if has_fairness:
        if not args.api_key:
            if args.env_api_key:
                args.api_key = get_env_api_key(args.model_provider)

        if not args.api_key:
            errors.append(
                f"API key required for fairness metrics: {[m for m in args.metrics if m in fairness_metrics]}"
            )
            errors.append(
                "Use --api-key YOUR_KEY or --env-api-key with environment variables"
            )

    if not args.model:
        args.model = setup_model_defaults(args.model_provider)

    if args.cpu_only:
        args.gpu = None
    elif args.gpu is None and not args.cpu_only:
        args.gpu = select_best_gpu()

    return errors


def generate_output_path(folder_path, metrics):
    """Generate output file path based on input folder and metrics"""
    folder_name = os.path.basename(os.path.abspath(folder_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_str = "_".join(sorted(metrics))

    filename = f"{folder_name}_{metrics_str}_{timestamp}.json"
    return os.path.join(os.path.dirname(os.path.abspath(folder_path)), filename)


def progress_callback(progress):
    """Simple progress callback for CLI"""
    percent = int(progress * 100)
    bar_length = 50
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(f"\rProgress: |{bar}| {percent}% Complete", end="", flush=True)


def main():
    """Main execution function"""
    args = parse_args()

    errors = validate_args(args)
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    if args.verbose:
        print("🔧 Configuration:")
        print(f"  Folder: {args.folder}")
        print(f"  Metrics: {', '.join(args.metrics)}")
        print(f"  Model Provider: {args.model_provider}")
        print(f"  Model: {args.model}")
        print(f"  GPU: {args.gpu if args.gpu is not None else 'CPU only'}")
        if args.api_key:
            masked_key = (
                f"{args.api_key[:10]}...{args.api_key[-4:]}"
                if len(args.api_key) > 14
                else "***"
            )
            print(f"  API Key: {masked_key}")
        print()

    if args.gpu is not None:
        try:
            gpu_info = get_gpu_info()
            if gpu_info and args.gpu < len(gpu_info):
                gpu = gpu_info[args.gpu]
                print(
                    f"🖥️  Using GPU {args.gpu}: {gpu['name']} ({gpu['memory_free']:.0f}MB free)"
                )
            else:
                print(f"⚠️  GPU {args.gpu} information not available")
        except Exception:
            print(f"⚠️  Could not get GPU {args.gpu} information")
    else:
        print("💻 Using CPU for processing")

    print()

    try:
        evaluator = ImageEvaluator(
            api_key=args.api_key,
            model_provider=args.model_provider,
            model_name=args.model,
            gpu_index=args.gpu,
        )
        print("✅ Evaluator initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return 1

    print("🚀 Starting image processing...")
    print(f"📁 Scanning folder: {args.folder}")

    try:
        results = evaluator.process_images(
            args.folder,
            args.metrics,
            progress_callback=progress_callback if args.verbose else None,
        )
        print()

        if "error" in results:
            print(f"❌ Processing error: {results['error']}")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        return 1

    if not args.output:
        args.output = generate_output_path(args.folder, args.metrics)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ Results saved successfully!")
        print(f"📄 Output file: {args.output}")

    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return 1

    print("\n📊 Processing Summary:")
    print(f"  Topic: {results['topic']}")
    print(f"  Configurations: {results['summary']['total_configs']}")
    print(f"  Total Images: {results['summary']['total_images']}")
    print(f"  Metrics Processed: {', '.join(args.metrics)}")
    print(f"  Output Size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")

    if args.verbose and results["configurations"]:
        print("\n🔍 Sample Results:")
        config_name = list(results["configurations"].keys())[0]
        config_data = results["configurations"][config_name]
        aggregates = config_data["aggregates"]

        print(f"  Configuration: {config_name}")
        print(f"  Images: {config_data['images']}")

        for metric in args.metrics:
            if metric == "clip_score" and "avg_clip_score" in aggregates:
                print(
                    f"  CLIP Score: {aggregates['avg_clip_score']:.4f} ± {aggregates.get('std_clip_score', 0):.4f}"
                )
            elif metric == "entropy_fairness" and "entropy_fairness" in aggregates:
                print(f"  Entropy Fairness: {aggregates['entropy_fairness']:.4f}")
            elif metric == "kl_fairness" and "kl_fairness" in aggregates:
                kl_val = aggregates["kl_fairness"]
                print(
                    f"  KL Fairness: {kl_val:.4f}"
                    if kl_val != float("inf")
                    else "  KL Fairness: ∞"
                )

    print("\n🎉 Processing completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

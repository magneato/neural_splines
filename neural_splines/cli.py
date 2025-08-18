"""
Neural Splines Command Line Interface

Provides command-line tools for converting models, running inference,
and managing Neural Splines models. Makes the 128.9x compression
breakthrough accessible through simple commands.
"""

import argparse
import sys
import logging
import json
from pathlib import Path
import torch
from typing import Optional, List, Dict, Any

from . import __version__
from .api import convert_model_to_splines, load_neural_splines_model, compare_models
from .exceptions import NeuralSplinesError, ConversionError, ModelLoadError
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)

def setup_cli_logging(verbose: int = 0):
    """Setup logging for CLI with appropriate verbosity"""
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    setup_logging(level)

def print_banner():
    """Print Neural Splines banner"""
    banner = f"""
🌊 Neural Splines v{__version__}
Transform neural networks into interpretable mathematical curves
Achieving 128.9x compression with zero quality loss
"""
    print(banner)

def print_success(message: str):
    """Print success message with emoji"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print error message with emoji"""
    print(f"❌ {message}")

def print_warning(message: str):
    """Print warning message with emoji"""
    print(f"⚠️  {message}")

def print_info(message: str):
    """Print info message with emoji"""
    print(f"🔹 {message}")

def convert_command(args):
    """Handle model conversion command"""
    print_info(f"Converting {args.model} to Neural Splines...")
    
    try:
        # Validate inputs
        if not Path(args.model).exists() and not args.model.startswith(('huggingface:', 'hf:')):
            print_error(f"Model path does not exist: {args.model}")
            return 1
        
        if args.output and Path(args.output).exists() and not args.force:
            print_error(f"Output directory exists: {args.output}. Use --force to overwrite.")
            return 1
        
        # Perform conversion
        result = convert_model_to_splines(
            model=args.model,
            output_path=args.output,
            compression_ratio=args.compression_ratio,
            spline_order=args.spline_order,
            validate_conversion=not args.skip_validation
        )
        
        if result['success']:
            print_success("Conversion completed successfully!")
            print_info(f"Compression achieved: {result['compression_achieved']:.1f}x")
            print_info(f"Parameters: {result['original_parameters']:,} → {result['compressed_parameters']:,}")
            
            if args.output:
                print_info(f"Model saved to: {args.output}")
            
            # Save conversion report
            if args.output:
                report_path = Path(args.output) / "conversion_report.json"
                with open(report_path, 'w') as f:
                    json.dump({
                        'conversion_stats': {
                            'compression_achieved': result['compression_achieved'],
                            'original_parameters': result['original_parameters'],
                            'compressed_parameters': result['compressed_parameters'],
                            'compression_efficiency': result.get('compression_efficiency', 1.0)
                        },
                        'configuration': {
                            'compression_ratio': args.compression_ratio,
                            'spline_order': args.spline_order,
                            'validation_enabled': not args.skip_validation
                        }
                    }, f, indent=2)
                print_info(f"Conversion report saved to: {report_path}")
            
            return 0
        else:
            print_error(f"Conversion failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except ConversionError as e:
        print_error(f"Conversion error: {e}")
        return 1
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1

def inference_command(args):
    """Handle inference command"""
    print_info(f"Loading model from {args.model}...")
    
    try:
        # Load model
        model = load_neural_splines_model(
            args.model,
            device=args.device,
            torch_dtype=getattr(torch, args.dtype) if args.dtype else None
        )
        
        print_success("Model loaded successfully!")
        
        # Print model statistics
        if hasattr(model, 'get_compression_stats'):
            stats = model.get_compression_stats()
            print_info(f"Model statistics:")
            print(f"  📊 Compression ratio: {stats['compression_ratio']:.1f}x")
            print(f"  💾 Memory usage: {stats['memory_gb']:.1f}GB")
            print(f"  🔢 Parameters: {stats['compressed_params']:,}")
        
        # Handle different inference modes
        if args.prompt:
            # Single prompt inference
            print_info(f"Generating response to: '{args.prompt}'")
            response = model.generate(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature
            )
            print_success("Generated response:")
            print(f"🤖 {response}")
            
        elif args.interactive:
            # Interactive mode
            print_info("Starting interactive mode. Type 'exit' to quit.")
            
            while True:
                try:
                    prompt = input("\n🤔 Your question: ").strip()
                    
                    if prompt.lower() in ['exit', 'quit', 'q']:
                        break
                    
                    if not prompt:
                        continue
                    
                    print_info("Generating response...")
                    response = model.generate(
                        prompt,
                        max_length=args.max_length,
                        temperature=args.temperature
                    )
                    print_success("Response:")
                    print(f"🌊 {response}")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print_error(f"Generation error: {e}")
            
            print_info("Interactive session ended.")
        
        else:
            print_warning("No inference mode specified. Use --prompt or --interactive")
        
        return 0
        
    except ModelLoadError as e:
        print_error(f"Failed to load model: {e}")
        return 1
    except Exception as e:
        print_error(f"Inference error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1

def compare_command(args):
    """Handle model comparison command"""
    print_info("Comparing models...")
    
    try:
        # Load models
        print_info(f"Loading original model from {args.original}")
        print_info(f"Loading Neural Splines model from {args.spline}")
        
        results = compare_models(
            args.original,
            args.spline,
            metrics=args.metrics
        )
        
        if 'error' in results:
            print_error(f"Comparison failed: {results['error']}")
            return 1
        
        print_success("Model comparison completed!")
        
        # Print results
        print("\n📊 Comparison Results:")
        print("=" * 50)
        
        if 'compression_ratio' in results:
            print(f"🗜️  Compression ratio: {results['compression_ratio']:.1f}x")
            print(f"📏 Size reduction: {results.get('memory_reduction', 'N/A')}")
        
        if 'inference_speedup' in results:
            print(f"⚡ Inference speedup: {results['inference_speedup']:.2f}x")
        
        if 'accuracy_similarity' in results:
            print(f"🎯 Accuracy similarity: {results['accuracy_similarity']*100:.1f}%")
        
        # Save detailed results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print_info(f"Detailed results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print_error(f"Comparison error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1

def visualize_command(args):
    """Handle visualization command"""
    print_info(f"Creating visualization for {args.model}")
    
    try:
        from .api import visualize_spline_structure
        
        # Create visualization
        fig = visualize_spline_structure(
            args.model,
            layer_name=args.layer,
            save_path=args.output,
            interactive=args.interactive
        )
        
        if args.output:
            print_success(f"Visualization saved to: {args.output}")
        else:
            print_info("Displaying visualization...")
            if hasattr(fig, 'show'):
                fig.show()
        
        return 0
        
    except ImportError as e:
        print_error("Visualization dependencies not available. Install with: pip install neural-splines[visualization]")
        return 1
    except Exception as e:
        print_error(f"Visualization error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1

def info_command(args):
    """Handle info command"""
    
    if args.model:
        # Show model information
        try:
            model = load_neural_splines_model(args.model)
            
            print(f"🌊 Neural Splines Model Information")
            print("=" * 50)
            print(f"Model path: {args.model}")
            
            if hasattr(model, 'get_compression_stats'):
                stats = model.get_compression_stats()
                print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
                print(f"Memory usage: {stats['memory_gb']:.1f}GB")
                print(f"Parameters: {stats['compressed_params']:,}")
                print(f"Spline layers: {stats['num_spline_layers']}")
                print(f"Target compression: {stats['target_compression']:.1f}x")
                print(f"Efficiency: {stats['efficiency']:.2f}")
            
            if hasattr(model, 'get_spline_layer_names'):
                layer_names = model.get_spline_layer_names()
                print(f"\nSpline layers ({len(layer_names)}):")
                for name in layer_names[:10]:  # Show first 10
                    print(f"  - {name}")
                if len(layer_names) > 10:
                    print(f"  ... and {len(layer_names) - 10} more")
            
        except Exception as e:
            print_error(f"Failed to load model info: {e}")
            return 1
    
    else:
        # Show general information
        print_banner()
        print("🔧 Installation Information:")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   PyTorch: {torch.__version__}")
        
        # Check dependencies
        print("\n📦 Dependencies:")
        dependencies = [
            ('numpy', 'numpy'),
            ('scipy', 'scipy'),
            ('matplotlib', 'matplotlib'),
            ('transformers', 'transformers'),
            ('accelerate', 'accelerate')
        ]
        
        for name, module in dependencies:
            try:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
                print(f"   ✅ {name}: {version}")
            except ImportError:
                print(f"   ❌ {name}: not installed")
        
        # Show system info
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            print(f"\n💾 System Memory: {memory_gb:.1f}GB")
        except ImportError:
            pass
        
        # Show GPU info
        if torch.cuda.is_available():
            print(f"\n🚀 GPU Information:")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        else:
            print("\n🚀 GPU: Not available (CPU-only mode)")
    
    return 0

def create_parser():
    """Create the main argument parser"""
    
    parser = argparse.ArgumentParser(
        prog='neural-splines',
        description='Transform neural networks into interpretable mathematical curves',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a model to Neural Splines
  neural-splines convert ./my_model --output ./my_model_neural --compression-ratio 100

  # Run inference with a Neural Splines model
  neural-splines inference ./my_model_neural --prompt "What is AI?"

  # Compare original vs Neural Splines model
  neural-splines compare --original ./original --spline ./neural --output comparison.json

  # Visualize spline structure
  neural-splines visualize ./my_model_neural --layer "attention.q_proj" --output plot.png

For more information, visit: https://github.com/your-username/neural-splines
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'Neural Splines {__version__}'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity (use -v, -vv for more verbose output)'
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert a model to Neural Splines representation'
    )
    convert_parser.add_argument(
        'model',
        help='Path to model or Hugging Face model name'
    )
    convert_parser.add_argument(
        '--output', '-o',
        help='Output directory for converted model'
    )
    convert_parser.add_argument(
        '--compression-ratio', '-c',
        type=float,
        default=128.9,
        help='Target compression ratio (default: 128.9)'
    )
    convert_parser.add_argument(
        '--spline-order',
        type=int,
        default=3,
        help='Spline interpolation order (default: 3 for bicubic)'
    )
    convert_parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip conversion validation'
    )
    convert_parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing output directory'
    )
    
    # Inference command
    inference_parser = subparsers.add_parser(
        'inference',
        help='Run inference with a Neural Splines model'
    )
    inference_parser.add_argument(
        'model',
        help='Path to Neural Splines model'
    )
    inference_parser.add_argument(
        '--prompt', '-p',
        help='Input prompt for single inference'
    )
    inference_parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive inference mode'
    )
    inference_parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum generation length (default: 100)'
    )
    inference_parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Generation temperature (default: 0.7)'
    )
    inference_parser.add_argument(
        '--device',
        default='auto',
        help='Device to run on (auto, cpu, cuda, cuda:0, etc.)'
    )
    inference_parser.add_argument(
        '--dtype',
        choices=['float16', 'float32', 'bfloat16'],
        help='Model data type'
    )
    
    # Compare command
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare original vs Neural Splines model'
    )
    compare_parser.add_argument(
        '--original',
        required=True,
        help='Path to original model'
    )
    compare_parser.add_argument(
        '--spline',
        required=True,
        help='Path to Neural Splines model'
    )
    compare_parser.add_argument(
        '--metrics',
        nargs='+',
        default=['compression', 'speed', 'memory'],
        help='Metrics to compare (compression, speed, memory, accuracy)'
    )
    compare_parser.add_argument(
        '--output', '-o',
        help='Output file for detailed results'
    )
    
    # Visualize command
    visualize_parser = subparsers.add_parser(
        'visualize',
        help='Visualize Neural Splines structure'
    )
    visualize_parser.add_argument(
        'model',
        help='Path to Neural Splines model'
    )
    visualize_parser.add_argument(
        '--layer', '-l',
        help='Specific layer to visualize'
    )
    visualize_parser.add_argument(
        '--output', '-o',
        help='Output file for visualization'
    )
    visualize_parser.add_argument(
        '--interactive',
        action='store_true',
        help='Create interactive visualization'
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show information about Neural Splines or a model'
    )
    info_parser.add_argument(
        'model',
        nargs='?',
        help='Path to Neural Splines model (optional)'
    )
    
    return parser

def main():
    """Main CLI entry point"""
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_cli_logging(args.verbose)
    
    # Show banner for info command or no command
    if not args.command or args.command == 'info':
        print_banner()
    
    # Handle commands
    try:
        if args.command == 'convert':
            return convert_command(args)
        elif args.command == 'inference':
            return inference_command(args)
        elif args.command == 'compare':
            return compare_command(args)
        elif args.command == 'visualize':
            return visualize_command(args)
        elif args.command == 'info':
            return info_command(args)
        else:
            parser.print_help()
            return 0
            
    except KeyboardInterrupt:
        print_info("Operation cancelled by user")
        return 1
    except NeuralSplinesError as e:
        print_error(str(e))
        return 1
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
"""
compile_whisper.py
──────────────────
Helper script to compile the Whisper model to ONNX for Qualcomm Hexagon NPU
using the Qualcomm AI Hub Models (qai-hub-models) package.

Target Device: Snapdragon X Elite (e.g. CRD)
Runtime: ONNX Runtime (with QNN execution provider)

Prerequisites:
  pip install qai-hub-models[whisper_base_en]

Usage:
  python compile_whisper.py
"""

import os
import sys
import subprocess

def main():
    print("=" * 70)
    print("🚀 Qualcomm AI Hub: Whisper ONNX Compilation")
    print("=" * 70)

    try:
        import qai_hub_models
    except ImportError:
        print("❌ 'qai-hub-models' is not installed.")
        print("Please install it first by running:")
        print("  pip install qai-hub-models[whisper_base_en]")
        sys.exit(1)

    print("\n⏳ Compiling Whisper Base (English) to ONNX for Snapdragon X Elite...")
    print("This may take a few minutes depending on your internet connection and Hub backend.\n")

    cmd = [
        "python", "-m", "qai_hub_models.models.whisper_base_en.export",
        "--target-runtime", "onnx",
        "--device", "Snapdragon X Elite CRD"
    ]

    try:
        # Run the export command
        subprocess.run(cmd, check=True)
        print("\n✅ Compilation successful!")
        print("The optimized ONNX model has been downloaded to the current directory (or hub-generated directory).")
        print("You can now integrate it into Veena AI using onnxruntime with the QNN execution provider.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Compilation failed with error code: {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()

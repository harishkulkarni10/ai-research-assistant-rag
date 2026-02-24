# vLLM server startup script
# Run this to start a vLLM inference server
#
# Usage:
#   python -m rag_arxiv_qa.src.inference.vllm.serve --model mistralai/Mistral-7B-Instruct-v0.2
#
# Or use the command directly:
#   python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --port 8000

import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Start vLLM inference server")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    
    args = parser.parse_args()
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--host", args.host,
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ]
    
    print(f"Starting vLLM server with model: {args.model}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"Command: {' '.join(cmd)}")
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

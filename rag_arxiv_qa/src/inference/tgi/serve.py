# TGI (Text Generation Inference) server startup script
# Run this to start a TGI inference server
#
# Note: TGI requires Docker or a pre-built TGI installation
#
# Docker usage:
#   docker run --gpus all -p 8080:80 ghcr.io/huggingface/text-generation-inference:latest \
#     --model-id mistralai/Mistral-7B-Instruct-v0.2
#
# Or if TGI is installed locally:
#   text-generation-server --model-id mistralai/Mistral-7B-Instruct-v0.2 --port 8080

import subprocess
import sys
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Start TGI inference server")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--use-docker", action="store_true", help="Use Docker to run TGI")
    
    args = parser.parse_args()
    
    if args.use_docker:
        cmd = [
            "docker", "run", "--gpus", "all", "-p", f"{args.port}:80",
            "ghcr.io/huggingface/text-generation-inference:latest",
            "--model-id", args.model
        ]
        print(f"Starting TGI server with Docker, model: {args.model}")
    else:
        tgi_cmd = os.getenv("TGI_CMD", "text-generation-server")
        cmd = [
            tgi_cmd,
            "--model-id", args.model,
            "--port", str(args.port),
        ]
        print(f"Starting TGI server locally, model: {args.model}")
    
    print(f"Server will be available at: http://localhost:{args.port}")
    print(f"Command: {' '.join(cmd)}")
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

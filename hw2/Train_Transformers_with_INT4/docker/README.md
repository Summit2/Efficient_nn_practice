# Docker training environment

The image uses the repository's tested PyTorch 1.12.1 and CUDA 11.3 stack. It
installs the Python dependencies, fetches CUTLASS 2.10, and builds all three
CUDA extensions used by the quantized BERT implementation.

Build the image and start the original CoLA/ BERT-base experiment from the
project root:

```bash
docker compose up --build -d
docker compose logs -f bert-int4
```

Training results are written to `SingleDivide/resultTime` and Hugging Face
downloads are cached in `hf-cache`. To stop the container, run
`docker compose down`.

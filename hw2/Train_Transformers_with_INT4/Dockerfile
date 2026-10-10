FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6" \
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade "pip<24" \
    && python -m pip install -r /tmp/requirements.txt

COPY Cuda_LSS_HQ /workspace/Cuda_LSS_HQ
COPY SingleDivide/transformersLocal /workspace/SingleDivide/transformersLocal

# The original setup.py files resolve CUTLASS at Cuda_LSS_HQ/cutlass/cutlass/include.
ARG CUTLASS_REF=feature/2.10/updates_before_tagging
RUN git clone --depth 1 --branch "${CUTLASS_REF}" \
        https://github.com/NVIDIA/cutlass.git Cuda_LSS_HQ/cutlass/cutlass \
    && cd Cuda_LSS_HQ/quantize_grad_weight_LSS+LSQ \
    && python setup.py install \
    && cd ../quantize_grad_input_LSS+LSQ \
    && python setup.py install \
    && cd ../quantize_forward_LSQ+HQ \
    && python setup_easy.py install

COPY SingleDivide /workspace/SingleDivide

RUN ln -sfn transformersLocal /workspace/SingleDivide/transformers \
    && cd /workspace/SingleDivide \
    && python -c "import accelerate, datasets, evaluate, transformers; print('Transformers', transformers.__version__)"

WORKDIR /workspace/SingleDivide
CMD ["bash", "-lc", "ln -sfn transformersLocal transformers && sed -i 's/\\r$//' task.sh && exec bash task.sh"]

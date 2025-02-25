ARG CUDA_IMAGE="12.6.0-devel-ubuntu24.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# # We need to set the host to 0.0.0.0 to allow outside access
# ENV HOST 0.0.0.0

RUN apt-get update && apt-get upgrade -y

# llama-cpp-python
RUN apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    python3.12-venv \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1

WORKDIR /app
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install depencencies
RUN /app/.venv/bin/pip install --upgrade cmake setuptools
# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86" FORCE_CMAKE=1 \
    /app/.venv/bin/pip install llama-cpp-python

RUN /app/.venv/bin/pip install langchain langchain-community langchain-nvidia-ai-endpoints langchain-openai

ENV PATH="/app/.venv/bin:${PATH}"
COPY manual-run_llama-cpp /app/

RUN mkdir -p /dataset_code-complete-iccad2023 /dataset_spec-to-rtl
COPY dataset_code-complete-iccad2023 /dataset_code-complete-iccad2023
COPY dataset_spec-to-rtl /dataset_spec-to-rtl

# RUN python --version
# RUN which python
# RUN ls -a
# RUN ls -a /


CMD [python3, "./manual-run.py"]

# syntax=docker/dockerfile:1

# Default upstream image for when not using buildx
ARG BASE_IMAGE=ghcr.io/neggles/tensorpods/base:cu121-torch210

# settings for apt and pip (inheritable by all images)
ARG DEBIAN_FRONTEND="noninteractive"
ARG DEBIAN_PRIORITY="critical"
ARG PIP_PREFER_BINARY="1"
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Build the base image.
FROM ${BASE_IMAGE} as neurosis

# Set shell
SHELL ["/bin/bash", "-ceuxo", "pipefail"]

# Inherit args from global
ARG DEBIAN_FRONTEND
ARG DEBIAN_PRIORITY
ARG PIP_PREFER_BINARY
ARG TORCH_CUDA_ARCH_LIST

# make pip STFU about being root
ENV PIP_ROOT_USER_ACTION="ignore"
ENV _PIP_LOCATIONS_NO_WARN_ON_MISMATCH="1"

# torch architecture list for from-source builds
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CUDA_MODULE_LOADING="LAZY"

# install bitsandbytes and xformers. these can be overridden with URLs
ARG XFORMERS_VERSION=xformers>=0.0.21
ARG BNB_VERSION=bitsandbytes>=0.42.0
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install \
        "${XFORMERS_VERSION}" \
        "${BNB_VERSION}"

# set workspace
WORKDIR /workspace

# add neurosis to workspace
COPY . /workspace/neurosis

# install neurosis
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    cd neurosis \
  && python -m pip install -e '.[all]'

# XDG dirs
ENV HOME="/workspace"

# wandb cache dirs
ENV WANDB_CACHE_DIR="${HOME}/.cache/wandb"
ENV WANDB_CONFIG_DIR="${HOME}/.config/wandb"
# huggingface cache dirs
ENV HF_HOME="${HOME}/.cache/huggingface"
ENV HF_HUB_CACHE="${HF_HOME}/hub"
ENV HF_ASSETS_CACHE="${HF_HOME}/assets"
ENV HF_DATASETS_CACHE="${HF_HOME}/datasets"

# set workdir to neurosis install dir
WORKDIR /workspace/neurosis

# this is set upstream but we may as well set it here too
CMD ["/bin/bash", "-l"]

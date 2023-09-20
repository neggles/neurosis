# docker-bake.hcl for stable-diffusion-webui
group "default" {
  targets = ["neurosis"]
}

variable "IMAGE_REGISTRY" {
  default = "ghcr.io"
}

variable "IMAGE_NAME" {
  default = "neggles/neurosis"
}

variable "BASE_IMAGE" {
  default = "ghcr.io/neggles/tensorpods/base"
}

variable "CUDA_VERSION" {
  default = "cu121"
}

variable "TORCH_VERSION" {
  default = "torch210"
}

variable "XFORMERS_VERSION" {
  default = "xformers==0.0.21"
}

variable "BNB_VERSION" {
  default = "bitsandbytes==0.41.1"
}

variable "TORCH_CUDA_ARCH_LIST" {
  default = "7.0;7.5;8.0;8.6;8.9;9.0+PTX"
}

# docker-metadata-action will populate this in GitHub Actions
target "docker-metadata-action" {}

# Shared amongst all containers
target "common" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  args = {
    BASE_IMAGE           = "${BASE_IMAGE}:${CUDA_VERSION}-${TORCH_VERSION}"
    TORCH_CUDA_ARCH_LIST = TORCH_CUDA_ARCH_LIST
  }
  platforms = ["linux/amd64"]
}

# Base trainer image
target "neurosis" {
  inherits = ["common", "docker-metadata-action"]
  target   = "neurosis"
  args = {
    XFORMERS_VERSION = XFORMERS_VERSION
    BNB_VERSION      = BNB_VERSION
  }
}

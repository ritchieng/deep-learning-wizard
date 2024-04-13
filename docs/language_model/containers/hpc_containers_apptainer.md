---
comments: true
---

# Containers

To run LLMs easily from laptops and desktops to the cloud, we will be introducing **Apptainer** (formerly Singularity) to achieve this where it has largely been adopted by the community to execute HPC workloads locally and in HPC data centers.

## Benefits of Apptainer

[Apptainer](https://apptainer.org/) has quite a few benefits such as:

- **Mobility of Compute**: It supports computational mobility through a single-file SIF container format.
- **Integration over Isolation**: It emphasizes integration, utilizing GPUs, high-speed networks, and parallel filesystems by default. In particular, it provides native support for running application containers using NVIDIA's CUDA or AMD's ROCm, enabling easy access to GPU-enabled machine learning frameworks like PyTorch, irrespective of the host operating system, provided the host has a driver and library installation for CUDA/ROCm.
- **Compatility with Docker**: It is 100% OCI compatible and aims for maximum compatibility with Docker.
- **Designed for Security**: It allows unprivileged users to use containers while preventing privilege escalation within the container and this means that users are the same within and outside the container.
- **Verifiable Reproducibility and Security**: It ensures trustworthy reproducibility and security through cryptographic signatures and an immutable container image format.

## Installing Apptainer

The following guide assumes a Linux distribution. We have also rigorously tested this on Windows WSL2 Ubuntu which works perfectly, so if you are on a Windows machine, this guide would work too. We are pleasantly surprised till this day, the sheer investment and improvement Windows made to bring WSL2 to a state where it is almost as good as a bare metal Linux distribution installation.

### Install Apptainer

All you need is to install [Apptainer](https://apptainer.org/docs/admin/latest/installation.html) to be able to leverage on this repository to work in containers with multiple environments (CPU/GPU) with any packages and OS independent of your host (local) machine.

### Install NVIDIA libraries

I advise you to use the [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software) to manage your drivers effectively without errors. It's a single bash script that allows you to install and upgrade your NVIDIA drivers.

Check CUDA toolkit installation via `nvcc -V`

### Set NVIDIA Container CLI Apptainer Paths (Deprecated)

Not necessary in Apptainer, only in Singularity.

To get nvidia-container-cli path
```
which nvidia-container-cli
```

To set
```
sudo singularity config global --set "nvidia-container-cli path" "/usr/bin/nvidia-container-cli"
```

To check
```
sudo singularity config global --get "nvidia-container-cli path"
```

## How to Use Apptainer in CPU/GPU Mode

### Option A: Transparent Image Container Workflow

This is recommended to build apptainer containers as it's transparent with the greatest reproducibility.

#### 1. Build container (`.sif`)

To build and test apptainer image container, simply run the following command in any of the folders:

```
bash build_sif.sh
```

#### 2. Run container

For CPU
```
apptainer shell apptainer_container_$VER_sandbox.sif
```

For GPU using NVIDIA Container CLI
```
apptainer shell --nv --nvccli apptainer_container_$VER_sandbox.sif
```

A quick way to test if you're able have everything good to run is to run `nvidia-smi` when you shell into the container. 

### Notes on GPU Availability

If you do not have a GPU on your host or local machine, the image will still be built and can be shared and used by machines with GPUs! The test for CUDA will fail on your local machine without GPU which will just spit out an error that you do not have a driver. There's nothing to worry about that.

### Option B: Blackbox

This is not recommended to build production apptainer containers but it is good for experimentation to determine the right configuration to put into your definition file for `Option A`.

Note, replace `$VER` with whatever version is in the specific folder.

#### 1. Build container (`.sif`)

To build and test apptainer image container, simply run the following command in any of the folders:
```
bash build_sif.sh
```

#### 2. Convert container to mutable sandbox

```
apptainer build --sandbox apptainer_container_$VER_sandbox apptainer_container_$VER.sif
``` 

#### 3. Shell into writable sandbox

You can install new packages and make persisting changes to the container in this step for experimentation.
```
apptainer shell --writable apptainer_container_$VER_sandbox
```

#### 4. Convert container to immutable image (`.sif`)

```
apptainer build --sandbox apptainer_container_$VER_sandbox.sif apptainer_container_$VER_img
``` 

## Basic Apptainer Commands Cheatsheet

This provides a list of useful commands.

```
# test apptainer
apptainer version

# pull cowsay container from docker
apptainer pull docker://sylabsio/lolcow

# build (generic that can also build from local definition files, it is like a swiss army knife, more flexible than pull)
apptainer build lolcow.sif docker://sylabsio/lolcow

# run option 1 requiring to shell into the container
apptainer shell lolcow.sif
cowsay moo

# run option 2 without requiring to shell into the container
apptainer exec lolcow.sif cowsay moo

# clean cache during builds
apptainer cache clean
```

## Available Containers & Definition Files

We maintain a repository of Apptainer recipes in this [repository](https://github.com/ritchieng/apptainer-recipes), feel free to clone/download it locally.

### GPU Containers

GPU containers can be found in `./containers/gpu` when you clone the above repository.

#### Ollama Workloads

##### Ollama General Workloads (Example: mistral)

- Go into container folder: `cd ./containers/gpu/ollama`
  - Run 1st session `apptainer shell --nv --nvccli apptainer_container_0.1.sif`
    - `ollama serve`
  - Run 2nd session (another window) `apptainer shell --nv --nvccli apptainer.1.sif`
    - `ollama run mistral`
    - You can now communicate with mistral model in your bash, or any other model you can pull on [ollama website](https://ollama.com/)

!!! info  "Model Choice"

    This runs a Mistral model as an example. You can run any other models by swapping out `mistral` reference above to any models on [Ollama's library](https://ollama.com/library).

##### Ollama General Workloads (Example: gemma)

- Go into container folder: `cd ./containers/gpu/ollama`
  - Run 1st session `apptainer shell --nv --nvccli apptainer_container_0.1.sif`
    - `ollama serve`
  - Run 2nd session (another window) `apptainer shell --nv --nvccli apptainer.1.sif`
    - `ollama run gemma:7b` or `ollama run gemma:2b`
    - You can now communicate with gemma model in your bash, or any other model you can pull on [ollama website](https://ollama.com/)

##### Ollama Multi-modal Workloads (Example: llava:7b-v1.6)

- Go into container folder: `cd ./containers/gpu/ollama`
  - Run 1st session `apptainer shell --nv --nvccli apptainer_container_0.1.sif`
    - `ollama serve`
  - Run 2nd session (another window) `apptainer shell --nv --nvccli apptainer.1.sif`
    - `ollama run llava:7b-v1.6`
    - You can now communicate with multi-modal llava model in your bash, or any other model you can pull on [ollama website](https://ollama.com/)

#### Ollama Embedding Workloads (Example: mxbai-embed-large)

- Go into container folder: `cd ./containers/gpu/ollama`
  - Run 1st session `apptainer shell apptainer_container_0.1.sif`
    - `ollama serve`
  - Run 2nd session (another window) `apptainer shell apptainer.1.sif`
    - `ollama pull mxbai-embed-large`
    - You can now communicate with the embedding model.

#### LLamaindex workloads

- Go into container folder: `cd ./containers/gpu/llamaindex`
  - Run 1st session (first window)`apptainer shell --nv --nvccli apptainer_container_0.1.sif`
    - `ollama serve`
  - Run 2nd session (second window) `apptainer shell --nv --nvccli apptainer_container_0.1.sif`
    - `ollama run mistral`
  - Run 3rd session (third window) `apptainer shell --nv --nvccli apptainer_container_0.1.sif`
    - `python`
    - `from llama_index.llms import Ollama`
    - `llm = Ollama(model="mistral")`
    - `response = llm.complete("What is Singapore")`
    - `print(response)`

!!! info  "Model Choice"

    This runs a Mistral model as an example. You can run any other models by swapping out `mistral` reference above to any models on [Ollama's library](https://ollama.com/library).

### CPU Containers

CPU containers can be found in `./containers/cpu` when you clone the above repository.

#### Math Workloads

- Go into container folder: `cd ./containers/cpu/math`
    - Run `apptainer shell apptainer_container_0.2.sif`

## Summary

We introduced Apptainer as a way to run CPU or GPU workloads that can scale from your desktops, to on-prem HPC data centers, or to cloud providers like Azure, AWS, and GCP. Subsequently, we introduced running LLMs within Apptainer offline locally with Gemma 2b/7b (Google) and Mistral 7b (Mistral AI) models. With portable container to run our code and models, we can now move to diving deep into LLMs, Multimodal Language Models, and Retrieval Augmented Generation (RAG).

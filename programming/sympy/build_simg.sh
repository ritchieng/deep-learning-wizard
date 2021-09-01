VER=0.1
IMAGE_NAME=./singularity_container_${VER}.simg

# Build
sudo singularity build ${IMAGE_NAME} ./singularity_container_${VER}.def

# Test pytorch
singularity exec --nv ${IMAGE_NAME} python -c "import torch;print('pytorch version: ' + torch.__version__)"

# Test pytorch gpu
singularity exec --nv ${IMAGE_NAME} python -c "import torch;print('pytorch cuda avail: ' + torch.cuda.is_available())"

echo image: ${IMAGE_NAME}
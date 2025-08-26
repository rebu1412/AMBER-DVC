NAME=videomamba
VERSION?=v5
COMMAND?=bash
OPTIONS?=

# GPU Configuration
GPUS?=all
ifeq ($(GPUS),none)
	GPUS_OPTION=
else
	GPUS_OPTION=--gpus=$(GPUS)
endif

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

.PHONY: all build stop run attach notebook info delete
# make all: Stops existing container, builds the image, and runs a new container with a bash shell
# make build: Builds the image.
# make stop: Stops and removes the container.
# make run: Runs a new container with a bash shell.
# make attach: Connects to a running container’s terminal.
# make notebook: Starts a Jupyter Lab server (http://localhost:8888).

info:
	@echo "Environment Variables:"
	@echo "NAME=$(NAME)"
	@echo "COMMAND=$(COMMAND)"
	@echo "OPTIONS=$(OPTIONS)"
	@echo "GPUS=$(GPUS)"
	@echo "GPUS_OPTION=$(GPUS_OPTION)"
	@echo "USER_ID=$(USER_ID)"
	@echo "GROUP_ID=$(GROUP_ID)"
	@echo "Current Directory=$(shell pwd)"

all: stop build run

# create a Docker image
# --target final
# docker build --target final -t $(NAME) . 2>&1 | tee ../build.log 
# docker build --target final -t $(NAME) .
build:
	docker build --target final -t $(NAME) .

# Stops and removes
stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

# Runs a new Docker container
# --rm: Automatically removes the container when it stops
# -dit: detached mode (-d), interactive terminal (-i), pseudo-TTY (-t)
# --net=host: Uses the host’s network stack, allowing the container to access the host’s network directly
# -v $(shell pwd):/workspace \

run:
	docker run --rm -dit \
		--net=host \
		--ipc=host \
		$(OPTIONS) \
		$(GPUS_OPTION) \
		-v $(shell pwd):/workspace \
		--user $(USER_ID):$(GROUP_ID) \
		--name=$(NAME) \
		$(NAME) \
		$(COMMAND)

	@docker attach $(NAME)

# connect container to terminal
attach:
	@docker attach $(NAME)

# 
notebook:
	@echo "Running docker run with NAME=$(NAME)"
	docker run --rm -it \
		--ipc=host \
		-p 8888:8888 \
		$(OPTIONS) \
		$(GPUS_OPTION) \
		-v $(shell pwd):/workspace \
		--user $(USER_ID):$(GROUP_ID) \
		--name=$(NAME)_notebook \
		$(NAME) \
		jupyter lab

del:
	docker rmi $(NAME)

save:
	@scp -r bombom2108@remote-server-ip:/home/user/project/runs ~/Downloads/

setup: docker image

docker:
	@echo "Setting-up Docker..."
	@sudo apt-get update
	@sudo apt-get install ca-certificates curl
	@sudo install -m 0755 -d /etc/apt/keyrings
	@sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
	@sudo chmod a+r /etc/apt/keyrings/docker.asc
	@echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu jammy stable" | \sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
	@sudo apt-get update
	@sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
	@echo "Checking Docker installation..."
	@docker --version > /dev/null 2>&1 && echo "Docker installed successfully: $$(docker --version)" || { echo "ERROR: Docker installation failed"; exit 1; }
	@sudo docker run hello-world > /dev/null 2>&1 && echo "Docker test run successful" || { echo "ERROR: Docker test run failed"; exit 1; }
	@echo "Adding user to Docker group..."
	@echo "Setting-up Nvidia-Container-Toolkit..."
	@curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
	@sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
	@sudo apt-get update 
	@sudo apt-get install -y nvidia-container-toolkit
	@echo "Configuring Docker for NVIDIA runtime..."
	@sudo nvidia-ctk runtime configure --runtime=docker > /dev/null 2>&1 && sudo systemctl restart docker && echo "Docker configured for NVIDIA runtime" || { echo "ERROR: Failed to configure NVIDIA runtime"; exit 1; }
	
image:	
	@echo "Pulling image with version $(VERSION)"
	@docker pull bombom2108/videomamba:$(VERSION)
	@docker tag bombom2108/videomamba:$(VERSION) videomamba
	@docker rmi bombom2108/videomamba:$(VERSION)

data:
	@echo "Downloading data..."
	@cd baseline/data/yc2/features/resnet_bn
	@bash download_yc2_tsn_features.sh
	@cd ../../..
	@echo "Checking downloaded data..."
	@[ -d "data/yc2/features" ] && [ -n "$$(ls -A data/yc2/features)" ] && echo "Data downloaded successfully" || { echo "ERROR: Data download failed or directory is empty"; exit 1; }
	@echo "Counting files in data/yc2/features..."
	@find data/yc2/features -type f | wc -l | xargs -I {} echo "Total number of files: {}"

# Stage 1
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel AS build
ARG UID=1000
ARG GID=1000
WORKDIR /workspace
COPY requirement.txt /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-11-jre && \
    apt-get install nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Tạo user và group
RUN groupadd -g ${GID} usergroup && useradd -m -u ${UID} -g usergroup user

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt

# Stage 2
FROM build AS pre
# RUN cd baseline/pdvc/ops && python setup.py build install && .. .. ..
# RUN pip install -e causal-conv1d/

# Stage 3
FROM pre AS final
# RUN pip install -e mamba/
ENV XDG_CACHE_HOME=/workspace/.cache
ENV HOME=/workspace

EXPOSE 8888



FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=23.11.0-0
ARG TARGETPLATFORM

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        tini \
        g++ \
        > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

WORKDIR /app
COPY . .

# 2. 创建环境并安装所有依赖
RUN \
    sed -i 's/build-backend = "poetry_dynamic_versioning.backend"/build-backend = "poetry.core.masonry.api"/' pyproject.toml && \
    conda create -n dpnegf python=3.10 -c conda-forge -y && \
    git clone https://github.com/deepmodeling/DeePTB.git && \
    conda run -n dpnegf pip install --upgrade pip setuptools wheel && \
    # [1] 强制拉取纯 CPU 版本的 PyTorch 2.1.1，极大地减小镜像体积并对齐底层接口
    conda run -n dpnegf pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu && \
    # [2] 强制使用 PyG 专属源拉取 torch-scatter，并使用 --only-binary=torch-scatter 彻底关闭源码编译。
    # 这样如果找不到精确匹配的 Wheel，它会立刻报错，而不是花 10 分钟编译出一个会引发崩溃的包。
    conda run -n dpnegf pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cpu.html --only-binary=torch-scatter && \
    # [3] 给本地仓库的安装加上 CPU 源保护，防止安装过程触发隐藏依赖，把刚才装好的 CPU 版 Torch 顶替成带 CUDA 的版本
    conda run -n dpnegf pip install ./DeePTB torch==2.1.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    conda run -n dpnegf pip install ./ torch==2.1.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    conda clean --all -y && \
    rm -rf /root/.cache/pip

# 3. 设置默认启动环境
RUN echo "conda activate dpnegf" >> ~/.bashrc

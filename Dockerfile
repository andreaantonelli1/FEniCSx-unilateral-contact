# Base image: FEniCSx v0.10.0 
FROM ghcr.io/fenics/dolfinx/dolfinx:v0.10.0

# !!Change this to your username!!
ARG USER=user 
ARG UID=1000

# remove potentially existing uid=1000 before creating new user
RUN id -nu ${UID} && userdel --force $(id -nu ${UID}) || true; \
    useradd -m -s /bin/bash ${USER} -u ${UID}

# Grant sudo rights (passwordless) + install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libxext6 \
        libxrender1 \
        python3-pip \
        wget \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && rm -rf /var/lib/apt/lists/*

# Create shared workspace
RUN mkdir -p /home/${USER}/shared && chown ${USER}:${USER} /home/${USER}/shared

# Install additional Python packages
# Note: dolfinx/basix/ufl/ffcx/petsc4py/mpi4py are provided by the base image.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Set defaults
USER ${USER}
WORKDIR /home/${USER}/shared

CMD ["/bin/bash", "-i"]
# Usa l'immagine base di Ubuntu
FROM ubuntu:22.04

# Imposta la variabile DEBIAN_FRONTEND per evitare prompt interattivi
ENV DEBIAN_FRONTEND=noninteractive

# Aggiorna i pacchetti e installa le dipendenze
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    gnupg \
    software-properties-common \
    vim \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Installa Zig (sostituisci la versione con quella desiderata)
ENV ZIG_VERSION=0.13.0
RUN wget https://ziglang.org/download/${ZIG_VERSION}/zig-linux-x86_64-${ZIG_VERSION}.tar.xz \
    && tar xf zig-linux-x86_64-${ZIG_VERSION}.tar.xz \
    && mv zig-linux-x86_64-${ZIG_VERSION} /opt/zig \
    && ln -s /opt/zig/zig /usr/local/bin/zig \
    && rm zig-linux-x86_64-${ZIG_VERSION}.tar.xz

# Imposta la directory di lavoro
WORKDIR /workspace

# Imposta l'utente non-root (opzionale ma consigliato)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Crea l'utente
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Cambia all'utente non-root
USER $USERNAME

# Comando di default
CMD [ "/bin/bash" ]


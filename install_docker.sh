#!/bin/bash
# Install Docker CE + Compose plugin on Ubuntu 22.04 (Jammy).
# Run with:  sudo bash install_docker.sh
set -e

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: run this script with sudo."
    exit 1
fi

REAL_USER="${SUDO_USER:-$USER}"

echo "==> Removing any old Docker packages..."
apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

echo "==> Installing prerequisites..."
apt-get update -q
apt-get install -y ca-certificates curl gnupg lsb-release

echo "==> Adding Docker official GPG key..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo "==> Adding Docker APT repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "==> Installing Docker CE + Compose plugin..."
apt-get update -q
apt-get install -y docker-ce docker-ce-cli containerd.io \
                   docker-buildx-plugin docker-compose-plugin

echo "==> Adding '$REAL_USER' to the docker group (no sudo needed for docker commands)..."
usermod -aG docker "$REAL_USER"

echo "==> Enabling and starting Docker..."
systemctl enable --now docker

echo ""
echo "Docker installation complete."
echo "docker --version: $(docker --version)"
echo "docker compose version: $(docker compose version)"
echo ""
echo "IMPORTANT: log out and back in (or run: newgrp docker) for group membership to take effect."

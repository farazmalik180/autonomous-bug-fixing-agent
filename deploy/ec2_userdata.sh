#!/bin/bash
# AWS EC2 Free Tier (Ubuntu 22.04 / 24.04 or Amazon Linux 2023) User-Data Bootstrap Script
set -euxo pipefail

# 1. Install Docker & Git
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y
    apt-get install -y git docker.io docker-compose-v2
    systemctl enable --now docker
elif command -v dnf >/dev/null 2>&1; then
    dnf update -y
    dnf install -y git docker
    systemctl enable --now docker
fi

# 2. Clone the repository
mkdir -p /opt/bugfix-agent
cd /opt/bugfix-agent
if [ ! -d ".git" ]; then
    git clone https://github.com/farazmalik180/autonomous-bug-fixing-agent.git .
else
    git pull origin main
fi

# 3. Create .env (replace GROQ_API_KEY below or set in UI)
cat <<EOF > .env
GROQ_API_KEY=${GROQ_API_KEY:-}
GROQ_MODEL=openai/gpt-oss-120b
SANDBOX_BACKEND=docker
EOF

# 4. Build the inner Docker sandbox image so tests run in isolated containers on EC2
docker build -t bugfix-agent-sandbox:latest src/sandbox || true

# 5. Build and run the Streamlit app container on port 80 and 8501
docker build -t bugfix-agent-web:latest .
docker rm -f bugfix-agent-web || true
docker run -d \
    --name bugfix-agent-web \
    --restart unless-stopped \
    -p 80:8501 \
    -p 8501:8501 \
    --env-file /opt/bugfix-agent/.env \
    -v /var/run/docker.sock:/var/run/docker.sock \
    bugfix-agent-web:latest

#!/usr/bin/env bash
# One-Click AWS CloudShell Deployment Script for Autonomous Bug-Fixing Agent
# Usage in AWS CloudShell:
#   GROQ_API_KEY="gsk_..." bash <(curl -sSL https://raw.githubusercontent.com/farazmalik180/autonomous-bug-fixing-agent/main/deploy/cloudshell_deploy.sh)

set -euo pipefail

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
SG_NAME="autonomous-bugfix-agent-sg"
INSTANCE_NAME="autonomous-bug-fixing-agent"

echo "=== 1/4 Configuring Security Group ($SG_NAME) in $REGION ==="
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)

if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    SG_ID=$(aws ec2 create-security-group \
        --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "Allow HTTP 80 and Streamlit 8501 for Autonomous Bug-Fixing Agent" \
        --vpc-id "$VPC_ID" \
        --query "GroupId" --output text)
fi

for PORT in 80 8501 22; do
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port "$PORT" \
        --cidr 0.0.0.0/0 >/dev/null 2>&1 || true
done
echo "Security Group ready: $SG_ID"

echo "=== 2/4 Finding latest Amazon Linux 2023 AMI in $REGION ==="
AMI_ID=$(aws ssm get-parameter \
    --region "$REGION" \
    --name "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64" \
    --query "Parameter.Value" --output text)
echo "Using AMI: $AMI_ID"

echo "=== 3/4 Building EC2 User-Data bootstrap script ==="
USER_DATA_FILE=$(mktemp)
cat <<EOF > "$USER_DATA_FILE"
#!/bin/bash
set -euxo pipefail
dnf update -y
dnf install -y git docker
systemctl enable --now docker

mkdir -p /opt/bugfix-agent
cd /opt/bugfix-agent
git clone https://github.com/farazmalik180/autonomous-bug-fixing-agent.git .

cat <<ENVEOF > .env
GROQ_API_KEY=${GROQ_API_KEY:-}
GROQ_MODEL=openai/gpt-oss-120b
SANDBOX_BACKEND=local
ENVEOF

docker build -t bugfix-agent-web:latest .
docker rm -f bugfix-agent-web || true
docker run -d \
    --name bugfix-agent-web \
    --restart unless-stopped \
    -p 80:8501 \
    -p 8501:8501 \
    --env-file /opt/bugfix-agent/.env \
    bugfix-agent-web:latest
EOF

echo "=== 4/4 Launching Free-Tier EC2 Instance ==="
INSTANCE_TYPE="t3.micro"
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --security-group-ids "$SG_ID" \
    --user-data "file://$USER_DATA_FILE" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --query "Instances[0].InstanceId" --output text 2>/dev/null || \
    aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "t2.micro" \
    --security-group-ids "$SG_ID" \
    --user-data "file://$USER_DATA_FILE" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
    --query "Instances[0].InstanceId" --output text)

rm -f "$USER_DATA_FILE"
echo "Launched instance: $INSTANCE_ID — waiting for Public IPv4..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo ""
echo "=================================================================="
echo "Instance $INSTANCE_ID is RUNNING!"
echo "Docker build finishes in ~2-3 minutes."
echo "Live App URL (Port 80):   http://$PUBLIC_IP"
echo "Live App URL (Port 8501): http://$PUBLIC_IP:8501"
echo "=================================================================="

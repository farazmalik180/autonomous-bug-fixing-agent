#!/usr/bin/env bash
# One-Command AWS CloudFront Free HTTPS Setup for Autonomous Bug-Fixing Agent
# Puts an Always-Free Amazon CloudFront HTTPS CDN (https://*.cloudfront.net) with WebSocket
# support in front of your running EC2 Streamlit instance.
#
# Usage in AWS CloudShell:
#   curl -sSL https://raw.githubusercontent.com/farazmalik180/autonomous-bug-fixing-agent/main/deploy/aws_https_cloudfront.sh | tr -d '\r' | bash

set -euo pipefail

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
SG_NAME="autonomous-bugfix-agent-sg"

echo "=== 1/3 Locating running EC2 instance in $REGION ==="
PUBLIC_DNS=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=tag:Name,Values=autonomous-bug-fixing-agent" "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].PublicDnsName" --output text)

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=tag:Name,Values=autonomous-bug-fixing-agent" "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

if [ -z "$PUBLIC_DNS" ] || [ "$PUBLIC_DNS" = "None" ]; then
    echo "Error: Could not find running 'autonomous-bug-fixing-agent' EC2 instance."
    exit 1
fi
echo "Found EC2 Origin: $PUBLIC_DNS ($PUBLIC_IP:8501)"

echo "=== 2/3 Ensuring Port 443 is open in Security Group ($SG_NAME) ==="
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" --filters "Name=group-name,Values=$SG_NAME" --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)
if [ -n "$SG_ID" ] && [ "$SG_ID" != "None" ]; then
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0 >/dev/null 2>&1 || true
fi

echo "=== 3/3 Creating Amazon CloudFront HTTPS Distribution (WebSocket-enabled) ==="
# AWS Managed CachePolicy: CachingDisabled (4135ea2d-6df8-44a3-9df3-4b5a84be39ad)
# AWS Managed OriginRequestPolicy: AllViewer (216adef6-5c7f-47e4-b989-5492eafa07d3)
CALLER_REF="bugfix-agent-$(date +%s)"
CF_CONFIG=$(mktemp)
cat <<EOF > "$CF_CONFIG"
{
  "CallerReference": "$CALLER_REF",
  "Comment": "Autonomous Bug-Fixing Agent Streamlit HTTPS (EC2 $PUBLIC_IP)",
  "Enabled": true,
  "Origins": {
    "Quantity": 1,
    "Items": [
      {
        "Id": "ec2-streamlit-origin",
        "DomainName": "$PUBLIC_DNS",
        "CustomOriginConfig": {
          "HTTPPort": 8501,
          "HTTPSPort": 443,
          "OriginProtocolPolicy": "http-only",
          "OriginSslProtocols": { "Quantity": 1, "Items": ["TLSv1.2"] },
          "OriginReadTimeout": 60,
          "OriginKeepaliveTimeout": 60
        }
      }
    ]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "ec2-streamlit-origin",
    "ViewerProtocolPolicy": "redirect-to-https",
    "AllowedMethods": {
      "Quantity": 7,
      "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
      "CachedMethods": { "Quantity": 2, "Items": ["GET", "HEAD"] }
    },
    "Compress": true,
    "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",
    "OriginRequestPolicyId": "216adef6-5c7f-47e4-b989-5492eafa07d3"
  },
  "PriceClass": "PriceClass_100"
}
EOF

CF_DOMAIN=$(aws cloudfront create-distribution \
    --distribution-config "file://$CF_CONFIG" \
    --query "Distribution.DomainName" --output text)
rm -f "$CF_CONFIG"

echo ""
echo "=================================================================="
echo "HTTPS CLOUDFRONT DISTRIBUTION CREATED!"
echo "Secure HTTPS URL: https://$CF_DOMAIN"
echo "(Note: CloudFront edge propagation takes ~2-3 minutes to finish deploying.)"
echo "=================================================================="

#!/usr/bin/env bash
# One-Command AWS Free-Tier Stack Provisioner (S3 + DynamoDB + Lambda Webhook + EC2 IAM Role)
# Usage in AWS CloudShell:
#   curl -sSL https://raw.githubusercontent.com/farazmalik180/autonomous-bug-fixing-agent/main/deploy/aws_free_tier_stack.sh | tr -d '\r' | bash

set -euo pipefail

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
BUCKET_NAME="bugfix-agent-reports-${ACCOUNT_ID}-${REGION}"
DDB_TABLE="BugFixAgentRuns"
LAMBDA_NAME="bugfix-agent-github-webhook"
ROLE_NAME="BugFixAgentCloudRole"
PROFILE_NAME="BugFixAgentEC2Profile"

echo "=== 1/5 Creating Amazon S3 Bucket ($BUCKET_NAME) ==="
if ! aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    if [ "$REGION" = "us-east-1" ]; then
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" >/dev/null
    else
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" \
            --create-bucket-configuration LocationConstraint="$REGION" >/dev/null
    fi
fi
echo "S3 Bucket ready: s3://$BUCKET_NAME"

echo "=== 2/5 Creating Amazon DynamoDB Table ($DDB_TABLE) ==="
if ! aws dynamodb describe-table --region "$REGION" --table-name "$DDB_TABLE" >/dev/null 2>&1; then
    aws dynamodb create-table \
        --region "$REGION" \
        --table-name "$DDB_TABLE" \
        --attribute-definitions AttributeName=run_id,AttributeType=S \
        --key-schema AttributeName=run_id,KeyType=HASH \
        --billing-mode PAY_PER_REQUEST >/dev/null
    aws dynamodb wait table-exists --region "$REGION" --table-name "$DDB_TABLE"
fi
echo "DynamoDB Table ready: $DDB_TABLE"

echo "=== 3/5 Configuring IAM Role ($ROLE_NAME) for EC2 & Lambda ==="
TRUST_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": ["ec2.amazonaws.com", "lambda.amazonaws.com"] },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
)
if ! aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST_POLICY" >/dev/null
fi

ACCESS_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::${BUCKET_NAME}", "arn:aws:s3:::${BUCKET_NAME}/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Scan", "dynamodb:Query"],
      "Resource": "arn:aws:dynamodb:${REGION}:${ACCOUNT_ID}:table/${DDB_TABLE}"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "*"
    }
  ]
}
EOF
)
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name "BugFixAgentS3DynamoAccess" --policy-document "$ACCESS_POLICY"
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query "Role.Arn" --output text)

if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1; then
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" || true
    sleep 8
fi

echo "=== 4/5 Attaching IAM Profile to Running EC2 Instance ==="
INSTANCE_ID=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=tag:Name,Values=autonomous-bug-fixing-agent" "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].InstanceId" --output text 2>/dev/null || true)

if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
    ASSOC_ID=$(aws ec2 describe-iam-instance-profile-associations \
        --region "$REGION" \
        --filters "Name=instance-id,Values=$INSTANCE_ID" \
        --query "IamInstanceProfileAssociations[0].AssociationId" --output text 2>/dev/null || true)
    if [ -z "$ASSOC_ID" ] || [ "$ASSOC_ID" = "None" ]; then
        aws ec2 associate-iam-instance-profile \
            --region "$REGION" \
            --instance-id "$INSTANCE_ID" \
            --iam-instance-profile Name="$PROFILE_NAME" >/dev/null || true
    fi
    echo "Attached IAM Role to EC2 instance: $INSTANCE_ID"
fi

echo "=== 5/5 Deploying AWS Lambda GitHub Webhook ($LAMBDA_NAME) ==="
TMP_DIR=$(mktemp -d)
curl -sSL https://raw.githubusercontent.com/farazmalik180/autonomous-bug-fixing-agent/main/deploy/lambda_webhook.py -o "$TMP_DIR/lambda_webhook.py"
(cd "$TMP_DIR" && zip -q function.zip lambda_webhook.py)

if aws lambda get-function --region "$REGION" --function-name "$LAMBDA_NAME" >/dev/null 2>&1; then
    aws lambda update-function-code \
        --region "$REGION" \
        --function-name "$LAMBDA_NAME" \
        --zip-file "fileb://$TMP_DIR/function.zip" >/dev/null
else
    aws lambda create-function \
        --region "$REGION" \
        --function-name "$LAMBDA_NAME" \
        --runtime python3.12 \
        --role "$ROLE_ARN" \
        --handler lambda_webhook.lambda_handler \
        --zip-file "fileb://$TMP_DIR/function.zip" \
        --timeout 15 \
        --environment "Variables={AWS_DYNAMODB_TABLE=$DDB_TABLE,AGENT_WEB_URL=http://98.89.45.201}" >/dev/null
fi

WEBHOOK_URL=$(aws lambda get-function-url-config --region "$REGION" --function-name "$LAMBDA_NAME" --query "FunctionUrl" --output text 2>/dev/null || \
    aws lambda create-function-url-config --region "$REGION" --function-name "$LAMBDA_NAME" --auth-type NONE --query "FunctionUrl" --output text)
aws lambda add-permission \
    --region "$REGION" \
    --function-name "$LAMBDA_NAME" \
    --statement-id "FunctionURLAllowPublicAccess" \
    --action "lambda:InvokeFunctionUrl" \
    --principal "*" \
    --function-url-auth-type "NONE" >/dev/null 2>&1 || true

rm -rf "$TMP_DIR"

echo ""
echo "=================================================================="
echo "AWS FREE-TIER STACK DEPLOYED SUCCESSFULLY!"
echo "1. S3 Report Bucket:      s3://$BUCKET_NAME"
echo "2. DynamoDB History Table: $DDB_TABLE"
echo "3. Lambda Webhook URL:    $WEBHOOK_URL"
echo "=================================================================="

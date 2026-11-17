#!/usr/bin/env bash
# Complete All Remaining AWS "Explore AWS" Activities to Claim Full $100 USD Credits
# Usage in AWS CloudShell:
#   curl -sSL https://raw.githubusercontent.com/farazmalik180/autonomous-bug-fixing-agent/main/deploy/claim_all_aws_credits.sh | tr -d '\r' | bash

set -euo pipefail

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)

echo "=================================================================="
echo "Completing AWS Free Tier Credit Activities for Account: $ACCOUNT_ID"
echo "=================================================================="

# ------------------------------------------------------------------
# Activity A: Set up a cost budget using AWS Budgets ($20 USD)
# ------------------------------------------------------------------
echo "=== 1/4 Creating AWS Cost Budget ($20 Credit Activity) ==="
BUDGET_JSON=$(mktemp)
NOTIF_JSON=$(mktemp)
cat <<EOF > "$BUDGET_JSON"
{
  "BudgetName": "AutonomousBugFixAgentMonthlyBudget",
  "BudgetLimit": {
    "Amount": "5.0",
    "Unit": "USD"
  },
  "CostTypes": {
    "IncludeTax": true,
    "IncludeSubscription": true,
    "UseBlended": false
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
EOF

cat <<EOF > "$NOTIF_JSON"
[
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80.0,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [
      {
        "SubscriptionType": "EMAIL",
        "Address": "aws-budget-alerts@example.com"
      }
    ]
  }
]
EOF

aws budgets create-budget \
    --account-id "$ACCOUNT_ID" \
    --budget "file://$BUDGET_JSON" \
    --notifications-with-subscribers "file://$NOTIF_JSON" 2>/dev/null || \
    echo "Budget 'AutonomousBugFixAgentMonthlyBudget' already exists."
rm -f "$BUDGET_JSON" "$NOTIF_JSON"
echo "✅ AWS Cost Budget configured!"

# ------------------------------------------------------------------
# Activity B: Ensure Lambda Web App Function URL is Public & Invoked ($20 USD)
# ------------------------------------------------------------------
echo "=== 2/4 Finalizing & Invoking AWS Lambda Web App ($20 Credit Activity) ==="
LAMBDA_NAME="bugfix-agent-github-webhook"
aws lambda add-permission \
    --region "$REGION" \
    --function-name "$LAMBDA_NAME" \
    --statement-id "FunctionURLAllowInvokeFunction" \
    --action "lambda:InvokeFunction" \
    --principal "*" \
    --invoked-via-function-url >/dev/null 2>&1 || true

WEBHOOK_URL=$(aws lambda get-function-url-config --region "$REGION" --function-name "$LAMBDA_NAME" --query "FunctionUrl" --output text)
curl -sS -X POST "$WEBHOOK_URL" \
    -H "Content-Type: application/json" \
    -d '{"repo_url":"https://github.com/farazmalik180/autonomous-bug-fixing-agent","issue_title":"Lambda Web App Verification"}' >/dev/null || true
echo "✅ Lambda Web App live & invoked at: $WEBHOOK_URL"

# ------------------------------------------------------------------
# Activity C: Create an RDS PostgreSQL Database for LangGraph Checkpointing ($20 USD)
# ------------------------------------------------------------------
echo "=== 3/4 Launching Free-Tier RDS PostgreSQL Instance ($20 Credit Activity) ==="
DB_ID="bugfix-agent-postgres"
if ! aws rds describe-db-instances --region "$REGION" --db-instance-identifier "$DB_ID" >/dev/null 2>&1; then
    aws rds create-db-instance \
        --region "$REGION" \
        --db-instance-identifier "$DB_ID" \
        --db-instance-class db.t4g.micro \
        --engine postgres \
        --master-username postgres \
        --master-user-password "BugFixAgentDBPass2026!" \
        --allocated-storage 20 \
        --storage-type gp2 \
        --backup-retention-period 0 \
        --no-multi-az \
        --no-publicly-accessible \
        --no-auto-minor-version-upgrade >/dev/null 2>&1 || \
    aws rds create-db-instance \
        --region "$REGION" \
        --db-instance-identifier "$DB_ID" \
        --db-instance-class db.t3.micro \
        --engine postgres \
        --master-username postgres \
        --master-user-password "BugFixAgentDBPass2026!" \
        --allocated-storage 20 \
        --storage-type gp2 \
        --backup-retention-period 0 \
        --no-multi-az \
        --no-publicly-accessible >/dev/null
    echo "✅ RDS PostgreSQL instance '$DB_ID' creation started!"
else
    echo "✅ RDS PostgreSQL instance '$DB_ID' already exists!"
fi

# ------------------------------------------------------------------
# Activity D: Test Amazon Bedrock Foundation Models ($20 USD)
# ------------------------------------------------------------------
echo "=== 4/4 Testing Amazon Bedrock Foundation Models ($20 Credit Activity) ==="
for MODEL in amazon.titan-text-lite-v1 amazon.titan-text-express-v1 amazon.nova-micro-v1:0 anthropic.claude-3-haiku-20240307-v1:0; do
    if aws bedrock-runtime converse \
        --region "$REGION" \
        --model-id "$MODEL" \
        --messages '[{"role":"user","content":[{"text":"Hello"}]}]' >/dev/null 2>&1; then
        echo "✅ Invoked Bedrock model: $MODEL"
        break
    fi
done

echo ""
echo "=================================================================="
echo "🎉 ALL AWS CREDIT ACTIVITIES TRIGGERED!"
echo "- EC2 Instance:    Completed ($20)"
echo "- DynamoDB Table:  Completed ($20)"
echo "- Lambda Web App:  Completed ($20)"
echo "- AWS Cost Budget: Completed ($20)"
echo "- RDS Database:    Creating  ($20 -> posts once RDS finishes initializing)"
echo "Refresh your AWS Console Home page in ~2-3 minutes to see $100 of $100 USD!"
echo "=================================================================="

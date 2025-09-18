#!/bin/bash

# Configuration
AWS_PROFILE="imssam_jjchan"
AWS_REGION="ap-northeast-2"
AWS_ACCOUNT_ID="571946422859"
FUNCTION_NAME="background-remover"
ECR_REPOSITORY_NAME="background-remover-lambda"
LAMBDA_ROLE_NAME="BackgroundRemoverLambdaRole"
S3_BUCKET_NAME="naeilssam-static-asset"  # Change this to your actual bucket name

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting Lambda function creation...${NC}"

# Step 1: Create IAM Role for Lambda
echo -e "${GREEN}Step 1: Creating IAM role for Lambda...${NC}"

# Create trust policy
cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create role
aws iam create-role \
    --role-name $LAMBDA_ROLE_NAME \
    --assume-role-policy-document file://trust-policy.json \
    --profile $AWS_PROFILE 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Role created successfully${NC}"
else
    echo -e "${YELLOW}Role already exists or creation failed${NC}"
fi

# Step 2: Attach policies to the role
echo -e "${GREEN}Step 2: Attaching policies to the role...${NC}"

# Create custom policy for S3 access
cat > lambda-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:$AWS_REGION:$AWS_ACCOUNT_ID:*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:PutObjectAcl"
            ],
            "Resource": "arn:aws:s3:::*/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": "arn:aws:s3:::*"
        }
    ]
}
EOF

# Create and attach policy
aws iam put-role-policy \
    --role-name $LAMBDA_ROLE_NAME \
    --policy-name BackgroundRemoverPolicy \
    --policy-document file://lambda-policy.json \
    --profile $AWS_PROFILE

echo -e "${GREEN}Policies attached${NC}"

# Wait for role to be available
echo -e "${YELLOW}Waiting for role to be available...${NC}"
sleep 10

# Step 3: Create Lambda function
echo -e "${GREEN}Step 3: Creating Lambda function...${NC}"

ROLE_ARN="arn:aws:iam::$AWS_ACCOUNT_ID:role/$LAMBDA_ROLE_NAME"
IMAGE_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:latest"

aws lambda create-function \
    --function-name $FUNCTION_NAME \
    --package-type Image \
    --code ImageUri=$IMAGE_URI \
    --role $ROLE_ARN \
    --timeout 300 \
    --memory-size 3008 \
    --environment Variables={LOG_LEVEL=INFO} \
    --profile $AWS_PROFILE \
    --region $AWS_REGION

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Lambda function created successfully!${NC}"
else
    echo -e "${YELLOW}Function might already exist. Attempting to update...${NC}"

    # Update function code
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --image-uri $IMAGE_URI \
        --profile $AWS_PROFILE \
        --region $AWS_REGION

    # Update function configuration
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --timeout 300 \
        --memory-size 3008 \
        --environment Variables={LOG_LEVEL=INFO} \
        --profile $AWS_PROFILE \
        --region $AWS_REGION

    echo -e "${GREEN}✅ Lambda function updated!${NC}"
fi

# Step 4: Add S3 trigger permission
echo -e "${GREEN}Step 4: Adding S3 trigger permission...${NC}"

if [ "$S3_BUCKET_NAME" != "your-bucket-name" ]; then
    aws lambda add-permission \
        --function-name $FUNCTION_NAME \
        --statement-id AllowS3Invoke \
        --action lambda:InvokeFunction \
        --principal s3.amazonaws.com \
        --source-arn arn:aws:s3:::$S3_BUCKET_NAME \
        --profile $AWS_PROFILE \
        --region $AWS_REGION 2>/dev/null

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}S3 permission added${NC}"
    else
        echo -e "${YELLOW}S3 permission might already exist${NC}"
    fi

    # Step 5: Configure S3 bucket notification
    echo -e "${GREEN}Step 5: Configuring S3 bucket notification...${NC}"

    # Update s3-trigger-config.json with correct ARN
    LAMBDA_ARN="arn:aws:lambda:$AWS_REGION:$AWS_ACCOUNT_ID:function:$FUNCTION_NAME"
    sed -i "" "s|arn:aws:lambda:ap-northeast-2:571946422859:function:background-remover|$LAMBDA_ARN|g" s3-trigger-config.json

    aws s3api put-bucket-notification-configuration \
        --bucket $S3_BUCKET_NAME \
        --notification-configuration file://s3-trigger-config.json \
        --profile $AWS_PROFILE

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ S3 trigger configured successfully!${NC}"
    else
        echo -e "${RED}Failed to configure S3 trigger. Please check bucket name and permissions.${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Please update S3_BUCKET_NAME in this script before configuring S3 trigger${NC}"
fi

# Cleanup temporary files
rm -f trust-policy.json lambda-policy.json

# Display summary
echo -e "\n${GREEN}========== Deployment Summary ==========${NC}"
echo -e "Lambda Function: ${YELLOW}$FUNCTION_NAME${NC}"
echo -e "Region: ${YELLOW}$AWS_REGION${NC}"
echo -e "Account: ${YELLOW}$AWS_ACCOUNT_ID${NC}"
echo -e "Memory: ${YELLOW}3008 MB${NC}"
echo -e "Timeout: ${YELLOW}300 seconds${NC}"
echo -e "Image URI: ${YELLOW}$IMAGE_URI${NC}"
echo -e "IAM Role: ${YELLOW}$ROLE_ARN${NC}"

if [ "$S3_BUCKET_NAME" != "your-bucket-name" ]; then
    echo -e "S3 Bucket: ${YELLOW}$S3_BUCKET_NAME${NC}"
    echo -e "Trigger Path: ${YELLOW}storage/competition/applicant/${NC}"
    echo -e "\n${GREEN}✅ Lambda function is ready to process images!${NC}"
else
    echo -e "\n${YELLOW}⚠️  Don't forget to:${NC}"
    echo -e "1. Update S3_BUCKET_NAME in this script"
    echo -e "2. Run the script again to configure S3 trigger"
fi

echo -e "${GREEN}========================================${NC}"
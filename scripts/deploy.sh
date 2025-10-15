#!/bin/bash

# Full deployment script: ECR + Lambda
# This script runs both deploy_to_ecr.sh and deploy_lambda.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${GREEN}Starting full deployment: ECR + Lambda...${NC}"
echo ""

# Step 1: Deploy to ECR
echo -e "${YELLOW}Step 1/2: Deploying to ECR...${NC}"
bash "$SCRIPT_DIR/deploy_to_ecr.sh"

if [ $? -ne 0 ]; then
    echo -e "${RED}ECR deployment failed. Aborting.${NC}"
    exit 1
fi

echo ""

# Step 2: Update Lambda
echo -e "${YELLOW}Step 2/2: Updating Lambda function...${NC}"
bash "$SCRIPT_DIR/deploy_lambda.sh"

if [ $? -ne 0 ]; then
    echo -e "${RED}Lambda update failed.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Full deployment completed successfully!${NC}"
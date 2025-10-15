#!/bin/bash

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Configuration - use environment variables with defaults
AWS_PROFILE="${AWS_PROFILE:-default}"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID}"
ECR_REPOSITORY_NAME="${ECR_REPOSITORY_NAME:-background-remover-lambda}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting deployment of Background Remover Lambda to ECR...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Get AWS Account ID
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${RED}Failed to get AWS Account ID. Please check your AWS credentials.${NC}"
    exit 1
fi

echo -e "${YELLOW}AWS Account ID: $AWS_ACCOUNT_ID${NC}"
echo -e "${YELLOW}Region: $AWS_REGION${NC}"
echo -e "${YELLOW}Repository: $ECR_REPOSITORY_NAME${NC}"

# Create ECR repository if it doesn't exist
echo -e "${GREEN}Creating ECR repository if it doesn't exist...${NC}"
aws ecr describe-repositories --repository-names $ECR_REPOSITORY_NAME --region $AWS_REGION --profile $AWS_PROFILE 2>/dev/null
if [ $? -ne 0 ]; then
    aws ecr create-repository --repository-name $ECR_REPOSITORY_NAME --region $AWS_REGION --profile $AWS_PROFILE
    echo -e "${GREEN}Repository created successfully${NC}"
else
    echo -e "${YELLOW}Repository already exists${NC}"
fi

# Get login token and login to ECR
echo -e "${GREEN}Logging into ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION --profile $AWS_PROFILE | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to login to ECR${NC}"
    exit 1
fi

# Build Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker buildx build --platform linux/amd64 --provenance=false -t $ECR_REPOSITORY_NAME:$IMAGE_TAG --load .

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build Docker image${NC}"
    exit 1
fi

# Tag image
echo -e "${GREEN}Tagging image...${NC}"
docker tag $ECR_REPOSITORY_NAME:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:$IMAGE_TAG

# Push image to ECR
echo -e "${GREEN}Pushing image to ECR...${NC}"
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:$IMAGE_TAG

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Image pushed to ECR successfully!${NC}"
    echo -e "${GREEN}Image URI: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME:$IMAGE_TAG${NC}"
else
    echo -e "${RED}Failed to push image to ECR${NC}"
    exit 1
fi

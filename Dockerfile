FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for image processing
RUN yum update -y && \
    yum install -y \
    libgomp \
    gcc \
    python3-devel \
    && yum clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy function code
COPY src/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src/background_remover.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]
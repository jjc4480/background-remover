FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies for image processing
RUN dnf update -y && \
    dnf install -y \
    libgomp \
    gcc \
    gcc-c++ \
    python3-devel \
    && dnf clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy function code
COPY src/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src/background_remover.py ${LAMBDA_TASK_ROOT}/
COPY src/mediapipe_face_detector.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]
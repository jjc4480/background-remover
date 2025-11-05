FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies for image processing
RUN dnf update -y && \
    dnf install -y \
    libgomp \
    gcc \
    gcc-c++ \
    python3-devel \
    mesa-libGL \
    && dnf clean all

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Pre-download BiRefNet model and store in /opt/models
# rembg downloads to ~/.u2net/, we copy to /opt for Lambda
RUN mkdir -p /opt/models && \
    python3 -c "from rembg import new_session; new_session('birefnet-general')" && \
    cp /root/.u2net/birefnet-general.onnx /opt/models/birefnet-general.onnx && \
    ls -lh /opt/models/

# MediaPipe 모델을 /opt/mediapipe에 미리 다운로드
RUN mkdir -p /opt/mediapipe && \
    python3 -c "import mediapipe as mp; \
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False); \
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0); \
    face_mesh.close(); pose.close()" && \
    cp -r /var/lang/lib/python3.12/site-packages/mediapipe /opt/mediapipe/ 2>/dev/null || true

# Copy function code
COPY src/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src/background_remover.py ${LAMBDA_TASK_ROOT}/
COPY src/mediapipe_face_detector.py ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]
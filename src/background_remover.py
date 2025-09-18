"""
배경 제거 핵심 로직 모듈
Lambda와 로컬 환경에서 공통으로 사용
"""
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove, new_session
import cv2
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """배경 제거 처리 클래스"""

    def __init__(self, model='birefnet-general'):
        """
        초기화

        Args:
            model: 사용할 모델명 (birefnet-general, u2net 등)
        """
        self.model = model
        self.session = None

    def _create_session(self):
        """모델 세션 생성"""
        if self.session is None:
            try:
                logger.info(f"Creating {self.model} session...")
                self.session = new_session(self.model)
                logger.info(f"Session created successfully: {type(self.session)}")
            except Exception as e:
                logger.warning(f"Failed to create {self.model} session: {e}")
                logger.info("Using default model...")
                self.session = None

    def remove_background(self, input_image):
        """
        이미지 배경 제거

        Args:
            input_image: PIL Image 객체

        Returns:
            PIL Image: 배경이 제거된 이미지 (RGBA)
        """
        # 세션 생성 (최초 1회)
        self._create_session()

        # 배경 제거
        logger.info("Removing background...")
        if self.session:
            output_image = remove(input_image, session=self.session)
        else:
            output_image = remove(input_image)

        # 후처리 적용
        output_image = self._apply_post_processing(output_image)

        return output_image

    def _apply_post_processing(self, image):
        """
        엣지 개선을 위한 후처리

        Args:
            image: PIL Image (RGBA)

        Returns:
            PIL Image: 후처리가 적용된 이미지
        """
        logger.info("Applying enhanced post-processing...")

        # numpy 배열로 변환
        img_array = np.array(image)

        if len(img_array.shape) < 3 or img_array.shape[2] != 4:
            logger.warning("Image is not RGBA format, skipping post-processing")
            return image

        # 알파 채널 추출
        alpha = img_array[:, :, 3].astype(np.float32) / 255.0

        # Trimap 생성 (확실한 전경/배경/불확실 영역 구분)
        trimap = np.zeros_like(alpha, dtype=np.uint8)
        trimap[alpha > 0.8] = 255  # 확실한 전경
        trimap[alpha < 0.1] = 0    # 확실한 배경
        trimap[(alpha >= 0.1) & (alpha <= 0.8)] = 128  # 불확실 영역

        # 가우시안 필터로 알파 채널 부드럽게
        alpha_smooth = gaussian_filter(alpha, sigma=1.5)

        # 엣지 보존 스무딩
        alpha_uint8 = (alpha_smooth * 255).astype(np.uint8)

        # Bilateral 필터로 엣지 보존하며 노이즈 제거
        alpha_bilateral = cv2.bilateralFilter(alpha_uint8, 9, 75, 75)

        # float로 다시 변환
        alpha_final = alpha_bilateral.astype(np.float32) / 255.0

        # 엣지 페더링 (자연스러운 전환)
        mask_binary = (alpha_final > 0.5).astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)

        if dist_transform.max() > 0:
            feather_radius = 3
            feather_mask = np.minimum(dist_transform / feather_radius, 1.0)
            alpha_final = alpha_final * feather_mask

        # 알파 채널 대비 향상 (더 깨끗한 엣지)
        alpha_final = np.clip(alpha_final * 1.2 - 0.1, 0, 1)

        # 알파 채널 업데이트
        img_array[:, :, 3] = (alpha_final * 255).astype(np.uint8)

        # PIL Image로 변환
        output_image = Image.fromarray(img_array, 'RGBA')

        # 색상 향상 (약간의 채도 증가)
        enhancer = ImageEnhance.Color(output_image)
        output_image = enhancer.enhance(1.05)

        return output_image

    def process_image(self, input_path, output_path):
        """
        파일 경로 기반 이미지 처리

        Args:
            input_path: 입력 이미지 경로
            output_path: 출력 이미지 경로
        """
        # 이미지 열기
        with Image.open(input_path) as input_image:
            # 배경 제거
            output_image = self.remove_background(input_image)

            # PNG로 저장 (투명도 유지)
            output_image.save(output_path, format='PNG')
            logger.info(f"Saved to: {output_path}")
#!/usr/bin/env python3
"""
로컬 환경에서 배경 제거를 테스트하는 스크립트
"""
import os
import sys
from pathlib import Path

# src 디렉토리를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from background_remover import BackgroundRemover
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_image(input_path, output_path=None, model='birefnet-general'):
    """
    단일 이미지 처리

    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로 (없으면 자동 생성)
        model: 사용할 모델명
    """
    input_path = Path(input_path)

    if not input_path.exists():
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return False

    # 출력 경로 생성
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_bg_removed.png"
    else:
        output_path = Path(output_path)

    logger.info(f"Processing: {input_path} -> {output_path}")

    try:
        # 배경 제거 처리
        bg_remover = BackgroundRemover(model=model)
        bg_remover.process_image(str(input_path), str(output_path))

        logger.info("처리 완료!")
        return True

    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        return False


def process_directory(input_dir, output_dir=None, model='birefnet-general'):
    """
    디렉토리 내 모든 이미지 처리

    Args:
        input_dir: 입력 디렉토리 경로
        output_dir: 출력 디렉토리 경로 (없으면 입력 디렉토리와 동일)
        model: 사용할 모델명
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        logger.error(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        return

    if output_dir is None:
        # 기본 출력 디렉토리는 bg_remove_test_output/
        output_dir = Path("bg_remove_test_output")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 지원되는 이미지 확장자
    supported_formats = {'.jpg', '.jpeg', '.png', '.webp'}

    # 이미지 파일 찾기
    image_files = [
        f for f in input_dir.iterdir()
        if f.suffix.lower() in supported_formats and '_bg_removed' not in f.name
    ]

    if not image_files:
        logger.info(f"처리할 이미지가 없습니다: {input_dir}")
        return

    logger.info(f"{len(image_files)}개 이미지 발견")

    # 배경 제거 처리
    bg_remover = BackgroundRemover(model=model)

    for img_file in image_files:
        output_file = output_dir / f"{img_file.stem}_bg_removed.png"

        try:
            logger.info(f"Processing: {img_file.name}")
            bg_remover.process_image(str(img_file), str(output_file))

        except Exception as e:
            logger.error(f"오류 발생 ({img_file.name}): {e}")
            continue

    logger.info("모든 이미지 처리 완료!")


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python local_runner.py <input_path> [output_path] [model]")
        print("")
        print("예시:")
        print("  python local_runner.py test.jpg")
        print("  python local_runner.py test.jpg output.png")
        print("  python local_runner.py bg_remove_test_images/")
        print("  python local_runner.py test.jpg output.png u2net")
        print("")
        print("지원 모델: birefnet-general (기본값), u2net, isnet-general")
        return

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    model = sys.argv[3] if len(sys.argv) > 3 else 'birefnet-general'

    input_path = Path(input_path)

    if input_path.is_file():
        # 단일 파일 처리
        process_single_image(input_path, output_path, model)
    elif input_path.is_dir():
        # 디렉토리 처리
        process_directory(input_path, output_path, model)
    else:
        logger.error(f"유효하지 않은 경로: {input_path}")


if __name__ == "__main__":
    main()
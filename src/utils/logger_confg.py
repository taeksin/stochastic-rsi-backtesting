"""
로깅 유틸리티 모듈

프로젝트 전반에서 사용하는 공통 로거 설정
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# UTF-8 출력 설정
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ✅ SUCCESS 로그 레벨 정의
SUCCESS_LEVEL_NUM = 25  # INFO(20)와 WARNING(30) 사이
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")


class ColoredFormatter(logging.Formatter):
    """컬러 출력과 줄바꿈을 지원하는 로그 포맷터"""

    COLORS = {
        'DEBUG': '\033[36m',    # 파랑
        'INFO': '\033[32m',     # 초록
        'SUCCESS': '\033[32m',  # 초록
        'WARNING': '\033[33m',  # 노랑
        'ERROR': '\033[31m',    # 빨강
        'CRITICAL': '\033[35m', # 보라
        'RESET': '\033[0m',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        return f"{timestamp} {color}[{record.levelname}]{reset} - {record.getMessage()}"


class Logger:
    """공통 로거 클래스 (Singleton)"""

    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._logger is None:
            self._setup_logger()

    def _setup_logger(self):
        self._logger = logging.getLogger('project_logger')
        self._logger.setLevel(logging.INFO)

        # 중복 핸들러 제거
        for h in self._logger.handlers[:]:
            self._logger.removeHandler(h)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        console_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(console_handler)

        # 파일 핸들러
        self._setup_file_handler()

    def _setup_file_handler(self):
        try:
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"log_{datetime.now().strftime('%Y-%m-%d')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            file_handler.setLevel(logging.DEBUG)
            self._logger.addHandler(file_handler)
        except Exception:
            pass

    # ✅ SUCCESS 레벨 메서드
    def success(self, message, *args, **kwargs):
        if self._logger.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._logger._log(SUCCESS_LEVEL_NUM, f"✅ {message}", args, **kwargs)

    # 기본 로깅 메서드들
    def debug(self, msg, *args, **kwargs): self._logger.debug(msg, *args, **kwargs)
    def info(self, msg, *args, **kwargs): self._logger.info(msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs): self._logger.warning(msg, *args, **kwargs)
    def error(self, msg, *args, exc_info=False, **kwargs): self._logger.error(msg, *args, exc_info=exc_info, **kwargs)
    def critical(self, msg, *args, exc_info=False, **kwargs): self._logger.critical(msg, *args, exc_info=exc_info, **kwargs)

    def progress(self, msg): self._logger.info(f"▶ {msg}")

    # ✅ 구분선 출력 (줄바꿈 없음)
    def newline(self):
        self._logger.info("=" * 50)

    # ✅ 빈 줄 출력 (줄바꿈만)
    def blankline(self):
        print()  # 단순 개행

    def set_level(self, level):
        """레벨 문자열로 로깅 수준 변경"""
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'SUCCESS': SUCCESS_LEVEL_NUM,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }
        lvl = levels.get(level.upper())
        if lvl:
            self._logger.setLevel(lvl)
            for h in self._logger.handlers:
                h.setLevel(lvl)


# ✅ 싱글톤 인스턴스 생성
logger = Logger()

# 간편 접근용 함수들
def get_logger(): return logger
def set_log_level(level): logger.set_level(level)
def log_debug(msg, *args, **kwargs): logger.debug(msg, *args, **kwargs)
def log_info(msg, *args, **kwargs): logger.info(msg, *args, **kwargs)
def log_success(msg): logger.success(msg)
def log_warning(msg, *args, **kwargs): logger.warning(msg, *args, **kwargs)
def log_error(msg, *args, exc_info=False, **kwargs): logger.error(msg, *args, exc_info=exc_info, **kwargs)
def log_critical(msg, *args, exc_info=False, **kwargs): logger.critical(msg, *args, exc_info=exc_info, **kwargs)
def log_blankline(): logger.blankline()


if __name__ == "__main__":
    test_log = get_logger()
    test_log.debug("디버그 메시지")
    test_log.info("일반 정보 메시지")
    test_log.success("성공 메시지 (초록색으로 표시됨)")
    test_log.warning("경고 메시지")
    test_log.blankline()
    test_log.error("에러 메시지")
    test_log.critical("치명적 오류 발생!")
    test_log.newline()

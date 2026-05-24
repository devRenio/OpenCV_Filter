# OpenCV_Filter

OpenCV와 Streamlit으로 만든 실시간 이미지 필터 웹 애플리케이션입니다.  
이미지 업로드 또는 웹캠으로 22가지 필터를 적용할 수 있습니다.

## 실행 방법

### 1. 사전 준비

- Python 3.9 이상
- 웹캠 실시간 모드를 사용할 경우 카메라 권한 허용

### 2. 의존성 설치

프로젝트 루트에서 가상환경을 만든 뒤 패키지를 설치합니다.

```bash
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

### 3. 앱 실행

```bash
streamlit run app.py
```

브라우저가 자동으로 열리지 않으면 터미널에 표시되는 주소(보통 `http://localhost:8501`)로 접속합니다.

### 4. 사용 방법

1. 사이드바에서 **입력 방식**을 선택합니다.
   - **이미지 업로드**: 파일을 올린 뒤 필터를 적용합니다.
   - **웹캠 (실시간)**: 카메라 영상에 필터를 실시간 적용합니다.
2. 원하는 **필터**와 **옵션**을 선택합니다.
3. 결과를 확인합니다.

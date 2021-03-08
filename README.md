# AI-face-processing
### 인공지능 안면인식 프로젝트
___

안면 인식을 통해 영상(input data) 속 인물과 기존에 등록되어 있는 인물(gallery)을 비교하여 신원을 확인하는 프로그램

```
AI-face-processing
│
├── README.md  
│
├── insightface_new
├── utils
│
├── face_enroll_by_id.py      - 갤러리 등록
├── face_enroll_by_name.py    - 갤러리 등록
├── vi.py                     - main code (Face Detection & Recognition with filtering)
├── vi_onevideo_v2.py         - IoU 이용 bounding box 중복(동일 객체 인식) 문제 처리
├── vi_eval.py                - 1:N 성능 평가 (FNIR @ FPIR of 0.01 or 0.1, DET Curve)
└── IJB-A Dataset.pdf         - 성능 평가를 위한 데이터셋과 Face Identification 평가 지표에 관한 설명
```
<br/><br/>

- Face Filtering - 정면 얼굴만 인식하도록 필터링
- Face Recognition
- Face Identification (1:N)
- Evaluation

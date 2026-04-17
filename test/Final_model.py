import pandas as pd
import torch
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================================================================
# 1. 설정 및 모델 초기화
# =============================================================================
class SentimentAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 한국어 감정 분석 모델
        model_name = "matthewburke/korean_sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # 부정적 키워드 리스트 (도메인 특화)
        self.negative_keywords = [
            "서버", "터짐", "렉", "버그", "팅김", "오류", "핵", "블랙홀",
            "환불", "망겜", "최적화", "발적화", "없데이트", "재탕",
            "현질", "가챠", "나락", "운영", "돈독", "욕설", "정신병자"
        ]

    def get_score(self, text):
        """
        단일 문장에 대한 긍정(1.0) ~ 부정(-1.0) 점수 산출
        """
        if not isinstance(text, str) or not text.strip():
            return 0.0
        
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        # probs[1] = 긍정 확률, probs[0] = 부정 확률
        return float(probs[1]) - float(probs[0])

    def analyze_sentiment_flow(self, text):
        """
        [NEW] 문장 간 감정 흐름(낙차) 분석
        Return: (max_contrast, sentence_scores)
        """
        if not text:
            return 0.0
        
        # 1. 문장 분리 (정규식 사용: . ? ! 뒤에 공백이 오면 자름)
        sentences = re.split(r'[.?!]\s+', text)
        sentences = [s for s in sentences if len(s.strip()) > 3] # 너무 짧은 문장 제외
        
        if len(sentences) < 2:
            return 0.0 # 문장이 하나면 낙차가 없음
            
        # 2. 각 문장별 점수 계산
        scores = [self.get_score(s) for s in sentences]
        
        # 3. 감정 낙차 계산 (최대 긍정 - 최대 부정)
        # 예: [+0.9, +0.8, -0.8] -> 0.9 - (-0.8) = 1.7 (매우 큼)
        if not scores:
            return 0.0
            
        max_score = max(scores)
        min_score = min(scores)
        contrast = max_score - min_score
        
        return contrast

    def has_keyword(self, text):
        """부정 키워드 포함 여부 확인"""
        for kw in self.negative_keywords:
            if kw in text:
                return True
        return False

# =============================================================================
# 2. 데이터 분석 로직 (종합 평가)
# =============================================================================
def analyze_review_row(row, analyzer):
    text = str(row.get('review', ''))
    voted_up = bool(row.get('voted_up', False))
    playtime_min = float(row.get('playtime_forever', 0))
    playtime_hour = playtime_min / 60.0

    # --- [Step 1] 기본 분석 ---
    # 1. 전체 문장 감정 점수 (Global Sentiment)
    ai_score = analyzer.get_score(text)
    # 2. 부정 키워드 유무
    has_neg_keyword = analyzer.has_keyword(text)
    # 3. [NEW] 감정 전이 낙차 (Sentiment Contrast)
    flow_contrast = analyzer.analyze_sentiment_flow(text)

    # --- [Step 2] 반어법(Sarcasm) 종합 판단 ---
    # Sarcasm Score: 반어법일 확률을 점수화 (0~100점 개념 아님, 높을수록 의심)
    # 기본 점수: 텍스트가 부정적인데 추천한 경우
    sarcasm_likelihood = 0.0
    
    # 조건 1: 맥락 불일치 (추천했는데 점수가 낮음)
    if voted_up and ai_score < -0.3:
        sarcasm_likelihood += 0.5
        
    # 조건 2: 감정 전이 낙차 (앞뒤 문장이 급격히 바뀜)
    # 낙차가 1.0 이상이면(예: +0.5 -> -0.5) 의심
    if flow_contrast > 1.0:
        sarcasm_likelihood += 0.4
        # 만약 낙차가 큰데, 부정 키워드까지 있다면 거의 확실
        if has_neg_keyword:
            sarcasm_likelihood += 0.3

    # 조건 3: 과도한 긍정 속에 숨겨진 키워드 (Ex: "환불하기 딱 좋은 갓겜")
    if voted_up and ai_score > 0.5 and has_neg_keyword:
        sarcasm_likelihood += 0.4

    # --- [Step 3] 최종 추천 여부 수정 (Decision Making) ---
    corrected_voted_up = voted_up
    decision_reason = "Original"

    # [Case A] 추천(Up) -> 비추천(Down) 변경 로직
    if voted_up:
        # 반어법 가능성이 높거나(0.7 이상), 점수가 너무 낮으면(-0.6 이하)
        if sarcasm_likelihood > 0.7 or ai_score < -0.6:
            corrected_voted_up = False
            decision_reason = "Detected Sarcasm/Negative"

    # [Case B] 비추천(Down) -> 추천(Up) 변경 로직
    elif not voted_up:
        # 키워드 없고, 점수 높고, 감정 낙차도 적은 경우 (순수 칭찬)
        if not has_neg_keyword and ai_score > 0.7 and flow_contrast < 0.8:
            corrected_voted_up = True
            decision_reason = "Detected Hidden Praise"

    # --- [Step 4] 신뢰도 등급 (Trust Level) ---
    trust_level = "NORMAL"
    if playtime_hour < 2.0:
        trust_level = "LOW_REFUND"
    elif playtime_hour < 10.0:
        trust_level = "LOW_EARLY"
        if voted_up and ai_score > 0.8 and flow_contrast < 0.5:
            trust_level = "LOW_HYPE" # 플레이 짧은데 너무 칭찬만 함
    elif playtime_hour >= 100.0:
        trust_level = "HIGH_VETERAN"
        if not voted_up:
            trust_level = "CRITICAL"

    return pd.Series([
        round(ai_score, 4),      # 전체 감정 점수
        round(flow_contrast, 4), # [NEW] 감정 낙차 점수
        round(sarcasm_likelihood, 2), # [NEW] 반어법 위험도
        decision_reason,         # 변경 사유
        corrected_voted_up,      # 최종 수정된 추천 여부
        trust_level              # 신뢰도
    ])

# =============================================================================
# 3. 메인 실행 함수
# =============================================================================
def process_steam_reviews(input_csv_path, output_csv_path):
    print(f"Loading data from {input_csv_path}...")
    
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print("Input file not found. Creating dummy data.")
        # 테스트 데이터 (반어법 케이스 포함)
        data = {
            "game_id": ["TEST", "TEST", "TEST", "TEST"],
            "review": [
                "그래픽은 진짜 좋은데 운영이 너무 쓰레기라 못해먹겠다.", # [반어법 의심] 낙차 큼
                "와 서버 관리 참 잘한다. 접속도 안되네 ^^", # [반어법 의심] 키워드 + 짧은 문장
                "이 게임 덕분에 여자친구가 생겼습니다. 거짓말입니다.", # [반어법] 긍정 -> 부정
                "진짜 재미없음. 절대 하지마세요." # [진짜 부정]
            ],
            "voted_up": [True, True, True, False], # 앞의 3개는 추천을 눌렀으나 내용은 부정적
            "playtime_forever": [600, 120, 50, 10],
            "timestamp_created": [0,0,0,0]
        }
        df = pd.DataFrame(data)

    analyzer = SentimentAnalyzer()
    print("Analyzing reviews (Comprehensive Mode)...")
    
    # 결과 컬럼 정의
    new_columns = [
        "ai_score", "flow_contrast", "sarcasm_score", 
        "decision_reason", "corrected_voted_up", "trust_level"
    ]
    
    # 분석 적용
    analysis_results = df.apply(lambda row: analyze_review_row(row, analyzer), axis=1)
    analysis_results.columns = new_columns
    
    # 병합 및 저장
    final_df = pd.concat([df, analysis_results], axis=1)
    final_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"Done! Saved to {output_csv_path}")
    print("\n--- 분석 결과 미리보기 ---")
    print(final_df[['review', 'voted_up', 'ai_score', 'flow_contrast', 'sarcasm_score', 'corrected_voted_up']].to_string())

if __name__ == "__main__":
    # 경로를 본인 환경에 맞게 수정하세요
    input_file = r"D:\code\steam_reviews\steam_reviews_clean.csv"
    process_steam_reviews(input_file, "steam_reviews_final_analysis.csv")
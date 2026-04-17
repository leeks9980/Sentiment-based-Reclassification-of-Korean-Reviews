import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# ============================================
# 1) 감정 분석 모델
# ============================================
sentiment_model_name = "nlp04/korean_sentiment_analysis_kcelectra"
sent_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# ============================================
# 2) 한국어 NLI 모델 (보조 신호용)
# ============================================
nli_model_name = "beomi/KcELECTRA-base"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

# ============================================
# 3) 의미 유사도 모델
# ============================================
embedding_model_name = "jhgan/ko-sroberta-sts"
embed_model = SentenceTransformer(embedding_model_name)

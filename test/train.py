import torch
import numpy as np
from sentence_transformers import util
import model

# ============================================
# 1) 감정 분석
# ============================================
def get_sentiment_prob(sentence):
    inputs = model.sent_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = model.sentiment_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    neg, pos = float(probs[0]), float(probs[1])
    return {"neg": neg, "pos": pos, "polarity": pos - neg}


# ============================================
# 2) 의미 유사도
# ============================================
def get_semantic_similarity(a, b):
    emb = model.embed_model.encode([a, b], convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb[0], emb[1]).item()
    return sim


# ============================================
# 3) 반어 표현 접속사 패턴
# ============================================
IRONY_CUE_WORDS = [
    "하지만", "그런데", "반대로", "오히려",
    "라고들 하지만", "라고 하는데", "인데", "인데도",
    "최악이라는데", "별로라는데",
]

def irony_pattern_score(sentence):
    for w in IRONY_CUE_WORDS:
        if w in sentence:
            return 1.0
    return 0.0


# ============================================
# 4) 의미 대비(반대 성향 단어 충돌)
# ============================================
POS_WORDS = ["좋다", "재미있다", "즐겁다", "최고", "만족", "재밌다", "훌륭"]
NEG_WORDS = ["최악", "별로", "끔찍", "형편없", "재앙", "실망", "불편"]

def lexical_contrast_score(sentence):
    pos_exist = any(w in sentence for w in POS_WORDS)
    neg_exist = any(w in sentence for w in NEG_WORDS)
    return 1.0 if (pos_exist and neg_exist) else 0.0


# ============================================
# 5) NLI 모순 (보조 신호)
# ============================================
def get_nli_contradiction(a, b):
    inputs = model.nli_tokenizer(f"{a} [SEP] {b}", return_tensors="pt", truncation=True)
    logits = model.nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    # ELECTRA 구조라 index가 달라 neutral 비중 큼 → contradiction을 작게 반영
    return float(probs[-1])  # 마지막 logit을 contradiction 의미로 사용


# ============================================
# 6) 메인 특징 추출
# ============================================
def extract_features(review):
    sents = [s.strip() for s in review.split('.') if s.strip()]

    sentiment_scores = [get_sentiment_prob(s) for s in sents]

    # 감정 polarity 대비
    if len(sents) >= 2:
        pols = [s["polarity"] for s in sentiment_scores]
        polarity_contrast = abs(pols[0] - pols[-1])
    else:
        polarity_contrast = 0.0

    # 의미 대비
    lexical_contrast = max(lexical_contrast_score(s) for s in sents)

    # 반어 패턴
    irony_pattern = max(irony_pattern_score(s) for s in sents)

    # 의미 유사도
    if len(sents) >= 2:
        semantic_sim = get_semantic_similarity(sents[0], sents[-1])
        semantic_contrast = 1 - semantic_sim
    else:
        semantic_contrast = 0.0

    # NLI (문장 2개일 때만)
    if len(sents) >= 2:
        nli_contra = get_nli_contradiction(sents[0], sents[-1])
    else:
        nli_contra = 0.0

    return {
        "sentiment": sentiment_scores,
        "polarity_contrast": polarity_contrast,
        "lexical_contrast": lexical_contrast,
        "irony_pattern": irony_pattern,
        "semantic_contrast": semantic_contrast,
        "nli": nli_contra,
    }

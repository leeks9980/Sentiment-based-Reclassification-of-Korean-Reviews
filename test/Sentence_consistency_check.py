import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer, util
import numpy as np

# ------------------------------
# 1) 감정 분석 모델 (이진: 긍정/부정)
# ------------------------------
sentiment_model_name = "nlp04/korean_sentiment_analysis_kcelectra"
sent_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

# ------------------------------
# 2) NLI 모델 (주의: 2-class 모델)
# ------------------------------
nli_model_name = "jhgan/ko-sroberta-multitask"  # 2-class 모델
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

# ------------------------------
# 3) 문장 임베딩 모델 (유사도 계산)
# ------------------------------
embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")


# ===========================================================
#   감정 분석
# ===========================================================
def get_sentiment_prob(sentence):
    inputs = sent_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = sentiment_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    neg, pos = float(probs[0]), float(probs[1])
    return (neg, pos)


# ===========================================================
#   NLI (모순 점수 계산)
#   ★ 이 모델은 2-class → 0: not contradiction, 1: contradiction
# ===========================================================
def nli_contradiction(a, b):
    inputs = nli_tokenizer(a, b, return_tensors="pt", truncation=True)
    outputs = nli_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    # ★ binary 모델이므로 contradiction = probs[1]
    return float(probs[1].item())


# ===========================================================
#   문장 간 의미 유사도
# ===========================================================
def semantic_similarity(sent_list):
    emb = embed_model.encode(sent_list, convert_to_tensor=True)
    sim_matrix = util.pytorch_cos_sim(emb, emb).cpu().numpy()
    return sim_matrix


# ===========================================================
#   과장 표현 점수 (단순 규칙 기반)
# ===========================================================
def overstatement(s):
    keywords = ["완전", "최고", "역대급", "절대", "무조건", "절망", "최악", "끔찍"]
    count = sum([1 for k in keywords if k in s])
    return count


# ===========================================================
#   전체 반어(사르카즘) 점수 계산
# ===========================================================
def sarcasm_score(review):
    sentences = [s.strip() for s in review.split('.') if len(s.strip()) > 0]

    if len(sentences) == 1:
        return 0.0, {"error": "한 문장만 있어서 비교 불가"}

    sentiment_probs = []
    contradictions = []
    overst = []
    sem_conflict = []

    for s in sentences:
        sentiment_probs.append(get_sentiment_prob(s))
        overst.append(overstatement(s))

    sim_matrix = semantic_similarity(sentences)

    # 문장 쌍 비교
    for i in range(len(sentences) - 1):
        c = nli_contradiction(sentences[i], sentences[i + 1])
        contradictions.append(c)

        # 의미 충돌 여부
        sim = sim_matrix[i][i + 1]
        sem_conflict.append(1 - sim)  # 유사도 낮으면 충돌 크다

    # ---------------------
    # 최종 반어 점수 계산
    # ---------------------
    sarcasm = (
        0.3 * max(contradictions)
        + 0.2 * max(sem_conflict)
        + 0.3 * (sum(overst) / len(overst))
        + 0.2 * int(max(overst) > 2)      # 과장 표현 여부
    )

    detail = {
        "sentences": sentences,
        "sentiment_prob": sentiment_probs,
        "contradiction_score": contradictions,
        "semantic_conflict": sem_conflict,
        "overstatement_score": overst,
        "similarity_matrix": sim_matrix.tolist(),
    }

    return sarcasm, detail


# ===========================================================
# 실행 예제
# ===========================================================
if __name__ == "__main__":
    review = "인생겜. 너만큼 오래한 게임은 없다. 너가 워너비야. 근데 렉만 고쳐줘."

    score, detail = sarcasm_score(review)

    print("=========== 반어 탐지 ===========")
    print(f"반어 점수 (0~1): {score}")
    print("세부 분석:")
    print(detail)

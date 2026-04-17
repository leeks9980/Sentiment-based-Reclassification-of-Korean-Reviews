from train import extract_features

def sarcasm_score(text):
    f = extract_features(text)

    # 가중치 최적화
    w = {
        "polarity_contrast": 0.35,
        "lexical_contrast": 0.25,
        "irony_pattern": 0.25,
        "semantic_contrast": 0.10,
        "nli": 0.05,
    }

    score = (
        w["polarity_contrast"] * f["polarity_contrast"] +
        w["lexical_contrast"] * f["lexical_contrast"] +
        w["irony_pattern"] * f["irony_pattern"] +
        w["semantic_contrast"] * f["semantic_contrast"] +
        w["nli"] * f["nli"]
    )

    score = round(min(score, 1.0), 4)

    return score, f


if __name__ == "__main__":
    text = "그럴 수 있어. 이런 날도 있는거지 뭐. X 항상 그렇지. 이런 날만 있는데 뭐. O"
    score, detail = sarcasm_score(text)
    print("Sarcasm Score:", score)
    print(detail)

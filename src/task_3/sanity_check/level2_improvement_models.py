from src.task_3.sanity_check.level1_improvement_models import fit_legacy, fit_logistic, fit_small

class TextOnlyPredictor:
    def __init__(self, predictor, market_features):
        self.predictor = predictor
        self.market_features = market_features

    def predict_proba(self, frame):
        masked = frame.copy()
        masked[self.market_features] = 0.
        return self.predictor.predict_proba(masked)

    def predict(self, frame):
        return (self.predict_proba(frame)[:, 1] >= .5).astype(int)

def fit_candidate(train, val, seed, candidate, market, native_text, padded_text, availability, original_config, epochs=None):
    text_only = candidate["representation"] == "T1"
    if text_only:
        train = train.copy()
        train[market] = 0.
        if val is not None:
            val = val.copy()
            val[market] = 0.
    if candidate["family"] == "logistic":
        predictor = fit_logistic(train, candidate, market, native_text, availability)
        best, ran, history = None, None, []
    elif candidate["family"] == "small_mlp":
        predictor, best, ran, history = fit_small(train, val, seed, candidate, market, native_text, availability, epochs=epochs)
    else:
        predictor, best, ran, history = fit_legacy(train, val, seed, market, padded_text, availability, original_config, selection="smooth")
    if text_only:
        predictor = TextOnlyPredictor(predictor, market)
    return predictor, best, ran, history

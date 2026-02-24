import sys
import numpy as np
from src.core.logger import logging
from src.core.exception import AppException
from src.app.monitoring.service_metrics import EXPLAINER_REQUEST_SUCCESS, EXPLAINER_REQUEST_FAILED

class ExplainerService:
    def __init__(self, explainer, model_booster, vectorizer):
        """
        Initializes an instance of LimeExplainer.

        Sets the class names for the explainer and initializes the LimeTextExplainer.
        Also initializes the model prediction attribute to None.
        """
        self.explainer = explainer
        self.booster = model_booster
        self.vectorizer = vectorizer

    def _get_prediction(self, text) -> np.ndarray:
        """
        Internal function to get class probability scores for lime explainer.
        """
        X = self.vectorizer.transform(text)
        prob = self.booster.inplace_predict(X)
    
        if len(prob.shape) == 1:
            prob = np.vstack([1 - prob, prob]).T
        return prob    
    
    def explain(self, text):
        try:
            explanation = self.explainer.explain_instance(
                text,
                self._get_prediction,
                num_features=10,
                num_samples=20
            )
            html_content = explanation.as_html()
            EXPLAINER_REQUEST_SUCCESS.inc()

            return html_content
        
        except Exception as e:
            EXPLAINER_REQUEST_FAILED.inc()
            logging.error(f"Explainer service failed: {e}", exc_info=True)
            raise AppException(e, sys)
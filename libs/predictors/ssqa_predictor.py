from typing import List, Dict, Any
from overrides import overrides
import json

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors import Predictor

from libs.tools.utils import find_closest_candidate


@Predictor.register("ssqa")
class SSQAPredictor(Predictor):
    """
    Predictor for the SSQA multiple choice problems.
    """

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """
        Override this method to output Chinese
        """
        return json.dumps(outputs, ensure_ascii=False) + "\n"

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        metadata = instance["metadata"]
        model_output = self._model.forward_on_instance(instance)

        # Select the most similar candidate among the multiple choices
        choices = metadata["choices"]
        best_span_str = model_output["best_span_str"]
        prediction = find_closest_candidate(
            best_span_str, list(choices.values()))
        correct = (prediction == metadata["answers"][0])

        output = {
            "id": metadata["id"],
            "context": metadata["context"],
            "question": metadata["question"],
            "choices": choices,
            "answer": metadata["answers"][0],
            "context_tokens": model_output["context_tokens"],
            "best_span": model_output["best_span"],
            "best_span_str": best_span_str,
            "prediction": prediction,
            "correct": correct
        }
        return sanitize(output)

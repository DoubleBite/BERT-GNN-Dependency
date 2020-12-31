"""

The reader architecture and part of the code is adpated from: 
https://github.com/allenai/allennlp-models/blob/master/allennlp_models/rc/dataset_readers/transformer_squad.py

TODO:
    + Make metadata and answers optional

"""

import json
import logging
from typing import Any, Dict, List, Tuple, Optional, Iterable
from overrides import overrides

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import MetadataField, TextField, SpanField
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.common.file_utils import cached_path, open_compressed


logger = logging.getLogger(__name__)


@DatasetReader.register("ssqa_span")
class SSQASpanReader(DatasetReader):
    """Reads a json-formatted SSQA dataset and returns a ``Dataset``.

    Fields:
        + ``question_with_context``: a ``TextField`` that contains the concatenation of question and context.
        + ``answer_span``: a ``SpanField`` into the ``question`` ``TextField`` denoting the answer.
        + ``context_span``: a ``SpanField`` into the ``question`` ``TextField`` denoting the context, i.e., the part of the text that potential answers can come from.
        + ``metadata``: a ``MetadataField`` that stores the instance's ID, the original question, the original context text, both of
            these in tokenized form, multiple choices, and the gold answer strings, accessible as ``metadata['id']``,
            ``metadata['question']``, ``metadata['context']``, ``metadata['question_tokens']``,
            ``metadata['context_tokens']``, ``metadata['choices']``, and ``metadata['answer']. This aims to make it easy when evaluation.

    We also support limiting the maximum length for the question. When the context+question is too long, we run a
    sliding window over the context and emit multiple instances for a single question. At training time, we only
    emit instances that contain a gold answer. At test time, we emit all instances. As a result, the per-instance
    metrics you get during training and evaluation don't correspond 100%.

    Args:
        + ``transformer_model_name`` : `str`, optional (default=`'bert-base-chinese'`)
            This reader chooses tokenizer and token indexer according to this setting.
        + ``length_limit`` : `int`, optional (default=`384`)
            We will make sure that the length of context+question never exceeds this many word pieces.
        + ``stride`` : `int`, optional (default=`128`)
            When context+question are too long for the length limit, we emit multiple instances for one question,
            where the context is shifted. This parameter specifies the overlap between the shifted context window. It
            is called "stride" instead of "overlap" because that's what it's called in the original huggingface
            implementation.
        + ``skip_invalid_examples``: `bool`, optional (default=`False`)
            If this is true, we will skip examples that don't have a gold answer. You should set this to True during
            training, and False any other time.
    """

    def __init__(
        self,
        transformer_model_name: str = "bert-base-chinese",
        length_limit: int = 384,
        stride: int = 128,
        skip_invalid_examples: bool = False,
        **kwargs
    ) -> None:

        super().__init__(**kwargs)
        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False
        )
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(transformer_model_name)}

        self.length_limit = length_limit
        self.stride = stride
        self.skip_invalid_examples = skip_invalid_examples

    @overrides
    def _read(self, file_path: str):

        logger.info("Reading file at %s", file_path)
        with open_compressed(cached_path(file_path), 'r') as dataset_file:
            dataset = json.load(dataset_file)

        logger.info("Reading the dataset")
        yielded_question_count = 0
        questions_with_more_than_one_instance = 0

        for example in dataset:
            qid = example["id"]
            context = example["passage"]  # We call it context here
            question = example["question"]
            choices = example["choices"]
            answer = example["answer"]
            answer_spans = example["answer_spans_char"]
            start_position, end_position = answer_spans[0]

            instances = self.make_instances(
                qid,
                context,
                question,
                choices,
                answer,
                start_position,
                end_position
            )
            instances_yielded = 0
            for instance in instances:
                yield instance
                instances_yielded += 1
            if instances_yielded > 1:
                questions_with_more_than_one_instance += 1
            yielded_question_count += 1

        if questions_with_more_than_one_instance > 0:
            logger.info(
                "%d (%.2f%%) questions have more than one instance",
                questions_with_more_than_one_instance,
                100 * questions_with_more_than_one_instance / yielded_question_count,
            )

    def make_instances(
        self,
        qid: str,
        context: str,
        question: str,
        choices: Dict[int, str],
        answer: Optional[str] = None,
        start_position: Optional[int] = None,
        end_position: Optional[int] = None,
    ) -> Iterable[Instance]:

        # Tokenize the context and question
        tokenized_context = self._tokenizer.tokenize(context)
        tokenized_question = self._tokenizer.tokenize(question)

        if answer is None:
            (start_position, end_position) = (-1, -1)

        space_for_context = (
            self.length_limit
            - len(tokenized_question)
            - len(self._tokenizer.sequence_pair_start_tokens)
            - len(self._tokenizer.sequence_pair_mid_tokens)
            - len(self._tokenizer.sequence_pair_end_tokens)
        )

        stride_start = 0
        while True:
            tokenized_context_window = tokenized_context[stride_start:]
            tokenized_context_window = tokenized_context_window[:space_for_context]

            window_token_answer_span = (
                start_position - stride_start,
                end_position - stride_start,
            )
            if any(i < 0 or i >= len(tokenized_context_window) for i in window_token_answer_span):
                # The answer is not contained in the window.
                window_token_answer_span = None

            if not self.skip_invalid_examples or window_token_answer_span is not None:
                metadata = {
                    "id": qid,
                    "context": context,
                    "question": question,
                    "choices": choices,
                    "answer": answer
                }
                instance = self.text_to_instance(
                    tokenized_question,
                    tokenized_context_window,
                    window_token_answer_span,
                    metadata,
                )
                yield instance

            stride_start += space_for_context
            if stride_start >= len(tokenized_context):
                break
            stride_start -= self.stride

    @overrides
    def text_to_instance(
        self,  # type: ignore
        tokenized_question: List[Token],
        tokenized_context: List[Token],
        token_answer_span: Optional[Tuple[int, int]],
        metadata: Dict[str, Any],
    ) -> Instance:

        # Make the question and context field.
        qc_field = TextField(
            self._tokenizer.add_special_tokens(
                tokenized_question, tokenized_context),
            self._token_indexers,
        )

        # Make the answer span
        start_of_context = (
            len(self._tokenizer.sequence_pair_start_tokens)
            + len(tokenized_question)
            + len(self._tokenizer.sequence_pair_mid_tokens)
        )

        if token_answer_span is not None:
            assert all(i >= 0 for i in token_answer_span)
            assert token_answer_span[0] <= token_answer_span[1]

            answer_span_field = SpanField(
                token_answer_span[0] + start_of_context,
                token_answer_span[1] + start_of_context,
                qc_field,
            )
        else:
            # We have to put in something even when we don't have an answer, so that this instance can be batched
            # together with other instances that have answers.
            answer_span_field = SpanField(-1, -1, qc_field)

        # make the context span, i.e., the span of text from which possible answers should be drawn
        context_span_field = SpanField(
            start_of_context, start_of_context +
            len(tokenized_context) - 1, qc_field
        )

        metadata.update({
            "question_tokens": tokenized_question,
            "context_tokens": tokenized_context,
        })

        fields = {
            "question_with_context": qc_field,
            "answer_span": answer_span_field,
            "context_span": context_span_field,
            "metadata": MetadataField(metadata)
        }

        return Instance(fields)

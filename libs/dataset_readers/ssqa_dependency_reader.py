"""
TODO
    + passage-to-passage graph connection

References:
    1. https://github.com/allenai/allennlp-models/blob/master/allennlp_models/rc/dataset_readers/transformer_squad.py
    2. https://github.com/allenai/allennlp-models/blob/master/allennlp_models/structured_prediction/dataset_readers/semantic_dependencies.py
    3. https://docs.allennlp.org/main/api/data/fields/adjacency_field/
"""

import json
import logging
from typing import Any, Dict, List, Tuple, Optional, Iterable
from overrides import overrides
from itertools import product
import random

from allennlp.data import Token, Instance, DatasetReader
from allennlp.data.fields import MetadataField, TextField, SpanField, AdjacencyField
from allennlp.data.token_indexers import PretrainedTransformerIndexer, PretrainedTransformerMismatchedIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.common.util import sanitize_wordpiece


logger = logging.getLogger(__name__)


@DatasetReader.register("ssqa_dependency")
class SSQADependencyReader(DatasetReader):
    """Reads a JSON-formatted ssqa dataset and returns a ``Dataset``.

    Note that because the data with dependnecy information is large, we need to use `lazy` mode to read the dataset.

    Fields:
        + ``question_with_context``: a ``TextField`` that contains the concatenation of question and contex.
        + ``answer_span``: a ``SpanField`` into the ``question`` ``TextField`` denoting the answer.
        + ``context_span``: a ``SpanField`` into the ``question`` ``TextField`` denoting the context, i.e., the part of the text that potential answers can come from.
        + ``metadata``: a ``MetadataField`` that stores the instance's ID, the original question, the original context text, both of
            these in tokenized form, and the gold answer strings, accessible as ``metadata['id']``,
            ``metadata['question']``, ``metadata['context']``, ``metadata['question_tokens']``,
            ``metadata['context_tokens']``, ``metadata['choices']``, and ``metadata['answer']. This aims to make it easy when evaluation.

    We also support limiting the maximum length for the question. When the context+question is too long, we run a
    sliding window over the context and emit multiple instances for a single question. At training time, we only
    emit instances that contain a gold answer. At test time, we emit all instances. As a result, the per-instance
    metrics you get during training and evaluation don't correspond 100%.

    Args:
        + ``transformer_model_name`` : `str`, optional (default=`'bert-base-cased'`)
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
        + ``max_query_length`` : `int`, optional (default=`64`)
            The maximum number of wordpieces dedicated to the question. If the question is longer than this, it will be
            truncated.
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
            "tokens": PretrainedTransformerMismatchedIndexer(transformer_model_name, max_length=512)}
        self.length_limit = length_limit
        self.stride = stride
        self.skip_invalid_examples = skip_invalid_examples

    @overrides
    def _read(self, file_path: str):

        logger.info("Reading file at %s", file_path)
        with open_compressed(cached_path(file_path), 'r') as dataset_file:
            dataset = json.load(dataset_file)

        # We dynamic shuffle the dataset every epoch because shuffling is disabled in lazy mode
        print("========================")
        print("Shuffle the dataset")
        random.shuffle(dataset)
        print("========================")

        # dataset = dataset[:1]
        logger.info("Reading the dataset")
        yielded_question_count = 0
        questions_with_more_than_one_instance = 0

        for example in dataset:
            qid = example["id"]
            passage_graph = example["passage_graph"]
            question_graph = example["question_graph"]
            choices = example["choices"]
            answer = example["answer"]
            spans = example["answer_spans_word"]
            start_position, end_position = spans[0]

            passage_nodes, passage_edges = passage_graph
            question_nodes, question_edges = question_graph

            instances = self.make_instances(
                qid,
                passage_nodes,
                passage_edges,
                question_nodes,
                question_edges,
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
        passage_nodes: Tuple[int, str, str],
        passage_edges: Tuple[int, int, str],
        question_nodes: Tuple[int, str, str],
        question_edges: Tuple[int, int, str],
        choices: Dict[int, str],
        answer: Optional[str],
        start_position: Optional[int],
        end_position: Optional[int],
    ) -> Iterable[Instance]:

        tokenized_context = [x[1] for x in passage_nodes]
        context = "".join(x for x in tokenized_context)
        tokenized_context = [Token(x) for x in tokenized_context]
        for idx, token in enumerate(tokenized_context):
            token.idx = idx
        tokenized_question = [x[1] for x in question_nodes]
        question = "".join(x for x in tokenized_question)
        tokenized_question = [Token(x) for x in tokenized_question]

        if answer is None:
            (start_position, end_position) = (-1, -1)

        # Stride over the context, making instances
        # Sequences are [CLS] question [SEP] [SEP] context [SEP], hence the - 4 for four special tokens.
        # This is technically not correct for anything but RoBERTa, but it does not affect the scores.
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
            passage_edges_window = self.filter_and_adjust_edges(
                passage_edges, stride_start, stride_start+space_for_context)

            window_token_answer_span = (
                start_position - stride_start,
                end_position - stride_start,
            )
            if any(i < 0 or i >= len(tokenized_context_window) for i in window_token_answer_span):
                # The answer is not contained in the window.
                window_token_answer_span = None

            if not self.skip_invalid_examples or window_token_answer_span is not None:
                additional_metadata = {
                    "id": qid,
                    "question": question,
                    "context": context,
                    "choices": choices,
                    "answers": [answer]
                }
                instance = self.text_to_instance(
                    tokenized_question,
                    question_edges,
                    tokenized_context_window,
                    passage_edges_window,
                    window_token_answer_span,
                    additional_metadata,
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
        question_edges: Tuple[int, int, str],
        tokenized_context: List[Token],
        passage_edges_window: Tuple[int, int, str],
        token_answer_span: Optional[Tuple[int, int]],
        metadata: Dict[str, Any] = None,
    ) -> Instance:
        fields = {}

        # Make the question and context field
        qp_field = TextField(
            self._tokenizer.add_special_tokens(
                tokenized_question, tokenized_context),
            self._token_indexers,
        )

        start_of_context = (
            len(self._tokenizer.sequence_pair_start_tokens)
            + len(tokenized_question)
            + len(self._tokenizer.sequence_pair_mid_tokens)
        )

        # Dependency edges
        # Something like: [(0, 2), [0, 5], ...]
        question_edge_indices = [(x[0]+len(self._tokenizer.sequence_pair_start_tokens),
                                  x[1]+len(self._tokenizer.sequence_pair_start_tokens))
                                 for x in question_edges]
        question_edge_labels = [x[2] for x in question_edges]
        passage_edge_indices = [(x[0]+start_of_context,
                                 x[1]+start_of_context)
                                for x in passage_edges_window]
        passage_edge_labels = [x[2] for x in passage_edges_window]
        # To save memory, do not pass the labels
        dependency_edge_field = AdjacencyField(
            question_edge_indices+passage_edge_indices, qp_field)
        # question_edge_indices+passage_edge_indices, qp_field, question_edge_labels+passage_edge_labels)

        # Interconnection edges between context and question nodes
        cross_edges = product(
            range(1, 1+len(tokenized_question)),
            range(start_of_context, start_of_context+len(tokenized_context)))
        cross_edges = list(cross_edges)
        cross_edge_field = AdjacencyField(
            # cross_edges, qp_field, ["cross"]*len(cross_edges))
            cross_edges, qp_field)

        # make the answer span
        if token_answer_span is not None:
            assert all(i >= 0 for i in token_answer_span)
            assert token_answer_span[0] <= token_answer_span[1]

            answer_span_field = SpanField(
                token_answer_span[0] + start_of_context,
                token_answer_span[1] + start_of_context,
                qp_field,
            )
        else:
            # We have to put in something even when we don't have an answer, so that this instance can be batched
            # together with other instances that have answers.
            answer_span_field = SpanField(-1, -1, qp_field)

        # make the context span, i.e., the span of text from which possible answers should be drawn
        context_span_field = SpanField(
            start_of_context, start_of_context +
            len(tokenized_context) - 1, qp_field
        )

        # Make the metadata
        metadata.update({
            "question_tokens": tokenized_question,
            "context_tokens": tokenized_context,
        })

        fields = {
            "question_with_context": qp_field,
            "edges": dependency_edge_field,
            "cross_edges": cross_edge_field,
            "answer_span": answer_span_field,
            "context_span": context_span_field,
            "metadata": MetadataField(metadata)
        }

        return Instance(fields)

    def filter_and_adjust_edges(self, edges, boundary_start, boundary_end):
        """Filter the edges outside the current sliding window and ajust the indices of the remaining ones.
        """
        new_edges = [x for x in edges if
                     (x[0] >= boundary_start and x[0] < boundary_end)
                     and (x[1] >= boundary_start and x[1] < boundary_end)]
        new_edges = [(x[0]-boundary_start, x[1]-boundary_start, x[2])
                     for x in new_edges]
        return new_edges

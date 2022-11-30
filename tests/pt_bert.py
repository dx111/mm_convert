import os
import time
import numpy as np
from tqdm import tqdm
from tests.mm_infer import MMInfer
import transformers
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, squad_convert_examples_to_features
from transformers.data.processors.squad import SquadResult, SquadV1Processor
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate


def get_results(features, example_indices, outputs):
    results = []
    for i, feature_index in enumerate(example_indices):
        eval_feature = features[feature_index.item()]
        unique_id = int(eval_feature.unique_id)
        output = [output[i].tolist() for output in outputs]
        start_logits, end_logits = output
        result = SquadResult(unique_id, start_logits, end_logits)
        results.append(result)
    return results

if __name__ == "__main__":
    model = MMInfer("pt_bert_model")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case = False)
    squad_processor = SquadV1Processor()
    examples = squad_processor.get_dev_examples("", filename="data/dev-v1.1.json")
    features, dataset = squad_convert_examples_to_features(
        examples = examples,
        tokenizer = tokenizer,
        max_seq_length = 128,
        doc_stride = 128,
        max_query_length = 64,
        is_training = False,
        return_dataset = "pt",
        threads = 4)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler = eval_sampler, batch_size = 1, drop_last = False)

    all_results = []
    for batch in tqdm(eval_dataloader, desc = "Evaluating"):
        batch = tuple(t for t in batch)
        outputs_np = model.predict([batch[0].numpy(), batch[1].numpy(), batch[2].numpy()])
        all_results.extend(get_results(features, batch[3], outputs_np))

    end_time = time.time()

    if not os.path.exists("output"):
        os.makedirs("output")

    output_prediction_file = "output/predictions.json"
    output_nbest_file = "output/nbest_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        20,  # n best size
        30,  # max answer length
        False,  # do lower case
        output_prediction_file,
        output_nbest_file,
        None,
        False,
        False,
        0.0,
        tokenizer
    )

    squad_acc = squad_evaluate(examples, predictions)

    print("SQUAD results: {}".format(squad_acc))
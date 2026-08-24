import argparse
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from utils.state_accuracy import (
    convert_xyxy_predictions_to_xywh,
    evaluate_state_accuracy,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate Anti-UAV410 State Accuracy from tracker result files.'
    )
    parser.add_argument(
        '--dataset-path',
        type=Path,
        required=True,
        help='Anti-UAV410 root directory containing test/ and/or val/.',
    )
    parser.add_argument(
        '--pred-path',
        type=Path,
        required=True,
        help='Directory containing one <sequence-name>.txt result per sequence.',
    )
    parser.add_argument('--split', choices=('test', 'val'), default='test')
    parser.add_argument(
        '--mode',
        type=int,
        choices=(1, 2),
        default=1,
        help='Prediction format: 1=XYWH, 2=XYXY.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('eval_details.txt'),
        help='Path for the per-sequence and overall score report.',
    )
    return parser.parse_args()


def _load_predictions(result_path: Path) -> Sequence:
    if not result_path.is_file():
        raise FileNotFoundError(f'Missing prediction file: {result_path}')

    text = result_path.read_text(encoding='utf-8').strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        values = np.loadtxt(io.StringIO(text.replace(',', ' ')), dtype=float)
        if values.size == 0:
            return []
        if values.ndim == 1:
            return [values.tolist()]
        return values.tolist()

    if isinstance(parsed, dict):
        if 'res' not in parsed:
            raise ValueError(f'JSON prediction file has no "res" field: {result_path}')
        parsed = parsed['res']
    if not isinstance(parsed, list):
        raise ValueError(f'Unsupported prediction JSON in {result_path}')
    return parsed


def evaluate_directory(
    dataset_path: Path,
    prediction_path: Path,
    split: str,
    mode: int,
    output_path: Path,
) -> float:
    split_path = dataset_path / split
    label_files = sorted(split_path.glob('*/IR_label.json'))
    if not label_files:
        raise FileNotFoundError(f'No IR_label.json files found under {split_path}')

    report_lines = []
    sequence_scores = []
    sequence_count = len(label_files)

    for sequence_index, label_path in enumerate(label_files, start=1):
        sequence_name = label_path.parent.name
        with label_path.open('r', encoding='utf-8') as file:
            labels = json.load(file)

        predictions = _load_predictions(prediction_path / f'{sequence_name}.txt')
        if mode == 2:
            predictions = convert_xyxy_predictions_to_xywh(predictions)

        score = evaluate_state_accuracy(predictions, labels)
        sequence_scores.append(score)
        line = (
            f'[{sequence_index:03d}/{sequence_count:03d}] '
            f'{sequence_name:>25} {"SA Score":>15}: {score:.04f}'
        )
        report_lines.append(line)
        print(line)

    overall_score = float(np.mean(sequence_scores))
    overall_line = f'[Overall] {"------":>25} {"SA Score":>15}: {overall_score:.04f}'
    report_lines.append(overall_line)
    print(overall_line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    return overall_score


def main() -> None:
    args = _parse_args()
    evaluate_directory(
        args.dataset_path,
        args.pred_path,
        args.split,
        args.mode,
        args.output,
    )


if __name__ == '__main__':
    main()

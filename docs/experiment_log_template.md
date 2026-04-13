# Experiment Log Template

Use one section per experiment. Keep this file factual and short. Its job is to preserve decisions, metrics, and observations you may need in a resume, report, or interview.

## Experiment Metadata

- Date:
- Dataset:
- Variant: `baseline / local_branch / transformer_branch / full_model`
- Goal:
- Hypothesis:

## Environment

- Python executable:
- Python version:
- torch / torchvision:
- Device:
- Random seed:

## Command

```bash
py -3.12 scripts/train.py \
  --data-root datasets \
  --dataset-name market1501 \
  --experiment-name market1501_baseline \
  --use-local-branch false \
  --use-transformer false \
  --use-fusion-gate false
```

## Outputs

- Output directory:
- Best checkpoint:
- Log file:
- Metrics file:

## Metrics

- Rank-1:
- Rank-5:
- Rank-10:
- mAP:

## Training Notes

- Did training converge normally?
- Any instability, OOM, or suspicious metric jump?
- Was the best epoch consistent with the loss curve?

## Retrieval Analysis

- Correct retrieval patterns:
- Failure cases:
- Typical confusion sources:

## Decision

- Keep this result as the main baseline? `yes / no`
- Next single change to test:
- Why that change is the next priority:

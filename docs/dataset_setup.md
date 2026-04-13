# Dataset Setup

Place benchmark datasets under a shared root such as `datasets/`.

## Market-1501

Expected layout:

```text
datasets/
  market1501/
    bounding_box_train/
    query/
    bounding_box_test/
```

File naming should follow the standard Market-1501 format, for example:

```text
0002_c1s1_000451_03.jpg
```

The loader extracts:

- person id from the first numeric field
- camera id from the `cX` field

## CUHK03-NP

Expected layout:

```text
datasets/
  cuhk03_np/
    bounding_box_train/
    query/
    bounding_box_test/
```

Use the standard CUHK03-NP split converted into Market-style filenames.

The current project uses the `detected` CUHK03-NP split by default because it is closer to the real detector-box setting. Prepare it from `archive.zip` and the Zhong et al. new-protocol split file:

```powershell
py -3.12 scripts\prepare_cuhk03_np.py --data-root datasets --variant detected --output-name cuhk03_np --jobs 24 --force
py -3.12 scripts\inspect_dataset.py --data-root datasets --dataset-name cuhk03_np
```

Expected detected split summary:

```json
{
  "train_images": 7365,
  "query_images": 1400,
  "gallery_images": 5332,
  "train_ids": 767
}
```

## MSMT17

Expected layout:

```text
datasets/
  msmt17/
    train/
    test/
    list_train.txt
    list_query.txt
    list_gallery.txt
```

The list files are expected to contain one image per line:

```text
0001/0001_00_0001.jpg 1
```

The loader uses:

- the provided label if present
- the second numeric token in the filename as camera id

## Sanity Check

Run:

```bash
python scripts/inspect_dataset.py --data-root datasets --dataset-name market1501
```

If the folder structure or filename pattern is wrong, the loader will raise a clear error before training starts.

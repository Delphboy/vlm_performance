# vlm_performance
How much VLM can you fit on your computer?

## Usage

1. Run `uv sync` to install dependencies.
2. Run `uv run main.py` to receive output.
3. Run `uv run main.py -h` to get usage options to change the model, split, and dataset directory.
4. The directory must have an `img` folder containing the images and a JSON file containing the splits, both in the root of the specified dataset directory.

## Example
`uv run main.py -m google/gemma-4-E2B-it -s dataset_coco.json -p datasets/coco`
# vlm_performance
How much VLM can you fit on your computer?

## Usage

1. Run `uv sync` to install dependencies.
2. Run `uv run main.py` to recieve output.
3. Run `uv run main.py modelname splitname` to change the default model and split

## Example
`uv run main.py google/gemma-4-E2B-it dataset_coco.json`
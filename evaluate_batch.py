import json
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

# Load the saved captions
with open("results.json", "r") as file:
    results = json.load(file)

references = {}
generated = {}

# Prepare captions for CIDEr evaluation
for index, item in enumerate(results):
    references[index] = item["reference_captions"]
    generated[index] = [item["generated_caption"]]

# Calculate CIDEr across all five images
cider = Cider()

score, individual_scores = cider.compute_score(
    references,
    generated
)

print("Images evaluated:", len(results))
print("Average CIDEr:", round(score, 4))


# Calculate SPICE across all five images
spice = Spice()

spice_score, individual_spice_scores = spice.compute_score(
    references,
    generated
)

print("Average SPICE:", round(spice_score, 4))
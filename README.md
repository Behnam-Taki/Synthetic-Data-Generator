# Synthetic-Data-Generator

## Sample Run

```python
import os

from model import SyntheticGeneratorModel


def get_unique_save_path(base_path):
    if not os.path.exists(base_path):
        return base_path

    base, ext = os.path.splitext(base_path)
    counter = 1
    new_path = f"{base}_{counter}{ext}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    return new_path


model_name = "model_#115"
model = SyntheticGeneratorModel(
    pos_list=['noun', 'adj', 'adv', 'verb'],
    pos_word_dist_count=2
)
print("Start derivation")
model.derivate(max_derivation_level=4, decay_factor=5)
print("Derivation done. Saving...")
save_path = get_unique_save_path(f"saved_models/{model_name}")
model.save(path=save_path)
print("Saved. Loading model...")
loaded_model = SyntheticGeneratorModel.load(path=save_path)
print("Model loaded.")
print(loaded_model.describe())
# print('\n\n\n\n')
print("Starting generation...")
for _ in range(10):
    print(loaded_model.root_document_distribution.generate_sample('root.#1.#1.#2.#1'), end='\n\n')

```

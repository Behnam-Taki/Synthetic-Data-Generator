# Synthetic-Data-Generator

## Sample Run

```python
from model import SyntheticGeneratorModel

model = SyntheticGeneratorModel(
    pos_list=['noun', 'adj', 'adv'],
    pos_word_dist_count=2
)
model.save()
loaded_model = SyntheticGeneratorModel.load()
print(loaded_model.describe())
print('\n\n\n\n')
for _ in range(10):
    print(loaded_model.root_text_distribution.generate_sample(), end='\n\n')
```
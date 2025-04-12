# Data Cleaning and Processing

`Preprocessing_v2`, `Preprocessing_v4`, and `Preprocessing_v5` represent successive iterations of the data cleaning and preprocessing pipeline applied to social media comments sourced from Facebook and Reddit. These scripts include steps such as:

- Converting emojis into descriptive text (e.g., 😡 → :angry_face:).

- Mapping Singlish and internet slang to standard English using a dictionary-based approach.

- Segmenting long or multi-entity comments into smaller, coherent chunks using GPT-4o-mini.

- Extracting the corresponding entity and emotion from each segment via LLM-assisted annotation.

- Removing noisy or low-information segments, such as those with fewer than four tokens or lacking key parts of speech (NOUN, PROPN, ADJ, VERB).

- Filtering out hallucinated or mismatched opinion phrases and remapping non-standard emotion labels to a controlled label set of seven emotions.

- In later versions, further refinement included manual vetting of segments, conversion of demojized text to proper descriptions, and removal of weakly labeled or ambiguous neutral samples.

Each version builds upon the last with more robust logic or stricter filtering to improve dataset quality for training downstream models.

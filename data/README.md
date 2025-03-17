# Data Cleaning and Processing for Sentiment Analysis

## Handling Slang and Acronyms
- Collected Singlish and internet slang from multiple sources.
- Mapped slangs to their meanings using a dictionary-based approach.
- Before LLM inference, appended slang definitions to the prompt to improve understanding.

## Annotating Sentiment and Emotion

### LLM-Based Annotation
- Designed n-shot prompting with structured examples to guide annotation.
- Multi-step process:
  - Step 1: Used an LLM to extract aspects and opinions from 2,572 Facebook comments (with existing entities and emotions).
  - Step 2: Reviewed results and refined prompts by incorporating slang definitions and colloquial terms.
  - Step 3: Applied the improved prompts to annotate 3,590 Reddit comments.
  - Final dataset: 8,744 annotated examples after structuring for flexibility. Note that the number of unique comments is smaller due to the duplications.

### Handling Multiple Entities Per Comment
- When a comment contained multiple entities, it was duplicated for each entity.
- Each duplicate was assigned its corresponding aspect, opinion, emotion, and sentiment.
- This format ensures compatibility with both classification models (`bert-base-uncased`) and Seq2Seq models.

## Standardizing Text
- Dropped rows where entity, aspect, opinion, sentiment, or emotion were missing or invalid (`None`, `"Unknown"`, `"N/A"`), ensuring data quality.
- Cleaned text by removing extra whitespace and special characters using regex.
- Converted emojis to text representations (e.g., `😃 → ":smiley:"`) using `emoji.demojize()`.

## Data Splitting for Model Training
- Split dataset into train-validation (80%) and test (20%), ensuring stratified sampling to preserve emotion distribution.

| emotion  | Count | Frequency |
|----------|-------|-----------|
| NEUTRAL  |  2009 | 23%       |
| DISGUST  |  1767 | 20%       |
| JOY      |  1648 | 18%       |
| SADNESS  |  1172 | 13%       |
| ANGER    |   976 | 11%       |
| FEAR     |   692 | 8%        |
| SURPRISE |   480 | 5%        |
| Total    |  8744 | 100%      |

- Further train-validation splitting for hyperparameter tuning is left to the individual user.

## Next steps
Codes to be added later.

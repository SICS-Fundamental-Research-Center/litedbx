GEN_FEAT_CANDIDATE_PROMPT = """
You are an expert in feature engineering for information retrieval and query optimization.

Your task is to identify *explicitly extractable, stable, and query-relevant* features from an unstructured source field.

Context:
- Source field modality: {MODALITY}
- Task description: {DESC}

Example data from the source field:
{SAMPLE_DATA}

Instructions:
Based on the context and examples above, propose a set of candidate features that can be reliably extracted from the source field and are likely to improve query accuracy or downstream retrieval performance.

Return a JSON object with a single key "features", whose value is a LIST  of feature configurations.

Each feature configuration MUST include the following fields:

- source_col:
  Always return the literal value "{SOURCE_COL}".

- target_col:
  It's the name of the new feature to be created.
  This should be a concise, self-explanatory snake_case name describing the feature to be extracted.

- prompt:
  A precise prompt that instructs an LLM how to extract this feature from a single data instance.
  The prompt should:
    • refer only to the source field content
    • clearly define the expected output format
    • avoid vague or subjective language

- feature_type:
  One of:
    • "boolean"
    • "numerical"

- domain:
  • If feature_type = "boolean", domain MUST be [false, true]
  • If feature_type = "numerical", domain MUST be a closed interval [a, b], where a ≤ b
    (use conservative bounds when exact limits are unknown)

Constraints:
- Propose only features that are derivable from the source field alone (no external data).
- Prefer features that are:
    • interpretable
    • robust across data variations
    • useful for filtering, ranking, or query rewriting
- Do NOT include free-text, categorical, or high-cardinality string features.
- Output valid JSON only. Do not include explanations or comments.
"""


PROMPTS = {
    "GEN_FEAT_CANDIDATE_PROMPT": GEN_FEAT_CANDIDATE_PROMPT,
}

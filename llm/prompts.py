GEN_FEAT_CANDIDATE_PROMPT = """
You are an expert in feature engineering for information retrieval and query optimization.

Your task is to identify *explicitly extractable, stable, and query-relevant* features from an unstructured source field.

Context:
- Source field modality: {MODALITY}
- Task description: {DESC}

Instructions:
Based on the context and examples above, propose a set of candidate features that can be reliably extracted from the source field and are likely to improve query accuracy or downstream retrieval performance.

Return a JSON object with a single key "features", whose value is a LIST  of feature configurations.

Each feature configuration MUST include the following fields:

- source_col:
  Always return the literal value "{SOURCE_COL}".

- source_modality:
  Always return the literal value "{MODALITY}".

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
  One of: ["bool", "float", "int"]


Constraints:
- Propose only features that are derivable from the source field alone (no external data).
- Propose at least 10 but no more than {FEATURE_BUDGET} distinct features.
- Prefer features that are:
    • interpretable
    • robust across data variations
    • useful for filtering, ranking, or query rewriting
- Do NOT include free-text, categorical, or high-cardinality string features.
- Output valid JSON only. Do not include explanations or comments.


We show the sampled data at below:
"""



SUGGEST_FEATURES_PROMPT = """
You are an expert in feature engineering for information retrieval and query optimization.

You will receive a list of candidate feature specifications that maybe helpful for improving query quality.

Your task is to select a subset of these candidate feature specifications.

Specifically, the input includes:
=== Input ===

1. Query Description: STRING
A description of the semantic query. You should select features that are most helpful to answer the query.

2. Candidate Feature Specifications: LIST 
These are provided as a LIST, where each item includes (a) `source_col`: the source feature name where the target feature is extracted (b) `target_col`: the name of the new feature to be extracted from `source_col`,  (c) `prompt`: the prompt to extract the `target_col`, and (c) feature_type: the type of `target_col` (boolean or numerical).

3. Selected Feature Specifications: LIST
These are provided as a LIST of feature specifications that have already been selected. You are suggested to select features that are complementary to these already selected features to reduce redundancy.

3. Selection Budget: INT
You should select exactly this number of features from the candidate list if candidates are sufficient, otherwise you should simply return all available candidates.


Here's the input for you:
=== Query Description ===
{semantic_desc}

=== Candidate Feature Specifications ===
{candidate_specs}

=== Selected Feature Specifications ===
{selected_specs}

=== Selection Budget ===
{selection_budget}


You should return a LIST of selected feature names. 
MAKE SURE:
- the returned feature name should be EXACTLY the same as the input feature name.
- return exactly {selection_budget} features from the candidate list if candidates are sufficient, otherwise you should simply return all available candidates.
- MAKE SURE NEVER return features that are already in the selected feature list.
"""





REFINE_FEATURE_SPACE_PROMPT = """
You are an expert in feature engineering for information retrieval and query optimization.

Your task is to refine a feature space based on classifier performance feedback.

=== Context ===
Task Description: {TASK_DESC}
Source Field: {SOURCE_FIELD}
Source Modality: {MODALITY}

=== Current Feature Space ===
{CURRENT_FEATURES}

=== Performance Feedback ===

1. Feature Importance (sorted by importance):
{FEATURE_IMPORTANCE}

2. F1 Score:
   - With current features: {F1_SCORE:.4f}
   - Improvement to previous features: {F1_IMPROVEMENT:.4f}

=== Instructions ===
Based on the feedback above, refine the feature space by:
1. **Adding new features** that could help distinguish the difficult cases shown above
2. **Removing low-importance features** that contribute little to prediction accuracy

Constraints:
- Propose at most {FEATURE_BUDGET} new features (for addition)
- You may suggest removing features with importance < 0.01
- Features should be extractable from the source field "{SOURCE_FIELD}" ({MODALITY})
- Output valid JSON only. Do not include explanations or comments.

Return a JSON object with two keys:
{{
  "to_add": [  // List of new feature specifications to add
    {{
      "source_col": "{SOURCE_FIELD}",
      "source_modality": "{MODALITY}",
      "target_col": "feature_name",
      "prompt": "extraction prompt",
      "feature_type": "bool" | "float" | "int"
    }}
  ],
  "to_remove": ["feature_name1", "feature_name2"]  // List of features to remove
}}

The bad cases (most uncertain misclassified samples) are provided at below:
"""


PROMPTS = {
    "GEN_FEAT_CANDIDATE_PROMPT": GEN_FEAT_CANDIDATE_PROMPT,
    "SUGGEST_FEATURES_PROMPT": SUGGEST_FEATURES_PROMPT,
    "REFINE_FEATURE_SPACE_PROMPT": REFINE_FEATURE_SPACE_PROMPT,
}

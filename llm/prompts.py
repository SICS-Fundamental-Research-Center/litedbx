GEN_FEAT_CANDIDATE_PROMPT = """
You are an expert in feature engineering for information retrieval and query optimization.

Your task is to identify *explicitly extractable, stable, and query-relevant* features from an unstructured source field using contrastive learning.

Context:
- Source field modality: {MODALITY}
- Task description: {DESC}

{SCHEMA_SAMPLE_SECTION}

{PREVIOUS_FEATURES_SECTION}

{PERFORMANCE_FEEDBACK_SECTION}

Instructions:
We provide you with contrastive examples: POSITIVE samples (labeled as satisfying the query) and NEGATIVE samples (labeled as not satisfying the query).
Your task is to identify features that can effectively distinguish between positive and negative samples.

{INSTRUCTIONS_SECTION}

Return a JSON object with two keys:
{{
  "to_add": [  // List of new feature specifications to add
    {{
      "source_col": "{SOURCE_COL}",
      "source_modality": "{MODALITY}",
      "target_col": "feature_name",
      "prompt": "extraction prompt",
      "feature_type": "bool" | "float" | "int"
    }}
  ],
  "to_remove": ["feature_name1", "feature_name2"]  // List of features to remove (empty if no previous features)
}}

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
- Propose at most {FEATURE_BUDGET} new features per iteration.
- Prefer features that are:
    • interpretable
    • robust across data variations
    • useful for filtering, ranking, or query rewriting
- Do NOT include free-text, categorical, or high-cardinality string features.
{CONSTRAINTS_ADDITIONAL}
- MUST include a feature that directly answer the task description
```
{DESC}
```
you should name it using a concise and self-explanatory name. It's prompt can be taken directly from the task description.
- Output valid JSON only. Do not include explanations or comments.

=== Contrastive Learning Data ===
We show the POSITIVE samples (satisfying the query) and NEGATIVE samples (not satisfying the query) at below:
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



AUGMENT_SIGMA_PROMPT = """
You are an expert in query optimization and data filtering.

Your task is to analyze a query and a dataset schema to suggest additional predicates that can narrow down the data range effectively.

=== Query Description ===
{query_desc}

=== Dataset Schema ===
The dataset contains the following columns with their value distributions:
{schema_info}

=== Instructions ===

Based on the query description and the dataset schema, suggest additional predicates (filters) that can help narrow down the data to find relevant records more efficiently.

For numerical columns, you can suggest predicates like:
- x > n or x >= n (lower bound)
- x < n or x <= n (upper bound)
- x == n (exact match)

For categorical columns, you can suggest:
- x == "CATEGORY" (exact match)
- x != "CATEGORY" (exclude specific value)

=== IMPORTANT SAFETY REMINDERS ===

1. ONLY return predicates when you are VERY CONFIDENT they will help narrow down the query-relevant data.
2. It is SAFE to return an EMPTY list if you are not confident.
3. A WRONG predicate will SIGNIFICANTLY DEGRADE the query quality by filtering out relevant data.
4. Be conservative - when in doubt, return nothing.
5. Only suggest predicates that align with the query intent.

=== Exact Match Indicator ===

Additionally, indicate whether applying the suggested predicates will EXACTLY match all desired records (no false positives, no false negatives).

Set `can_exact_match` to True ONLY if:
- The predicates are sufficient to precisely identify ALL relevant records
- The predicates will NOT include any irrelevant records
- You have VERY HIGH confidence in this assessment

Otherwise, set `can_exact_match` to False.

=== Response Format ===

Return a JSON object with the following structure:
{{
  "value": [  // List of suggested predicates (can be empty)
    {{
      "field": "column_name",
      "op": ">" | ">=" | "<" | "<=" | "==" | "!=",
      "value": numeric_value | "string_value" | true | false
    }}
  ],
  "can_exact_match": true | false  // Whether predicates exactly match desired records
}}

Examples:
- For numerical: {{"field": "price", "op": ">=", "value": 100}}
- For categorical: {{"field": "category", "op": "==", "value": "electronics"}}
- For boolean: {{"field": "is_active", "op": "==", "value": true}}

Output valid JSON only. Do not include explanations or comments.
"""


PROMPTS = {
    "GEN_FEAT_CANDIDATE_PROMPT": GEN_FEAT_CANDIDATE_PROMPT,
    "SUGGEST_FEATURES_PROMPT": SUGGEST_FEATURES_PROMPT,
    "AUGMENT_SIGMA_PROMPT": AUGMENT_SIGMA_PROMPT,
}

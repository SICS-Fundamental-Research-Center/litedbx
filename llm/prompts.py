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



IDENTIFY_RELEVANT_FIELDS_PROMPT = """
You are an expert in query optimization and data analysis.

Your task is to identify and semantically group the fields in a dataset that are relevant to answering a query.

=== Query Description ===
{query_desc}

=== Dataset Schema ===
The dataset contains the following columns with their value distributions:
{schema_info}

=== Instructions ===

Based on the query description, identify ALL fields (columns) that are relevant to answering the query and group them by semantic meaning AND value range compatibility.

A field is relevant if:
- It contains information directly related to the query intent
- It can help distinguish between query-relevant and query-irrelevant records
- It is mentioned or implied by the query description

Semantic Grouping with Value Range Compatibility:

Group fields that BOTH:
1. **Share semantic meaning** - represent the same concept or attribute (e.g., baseColour, colour1, colour2 all represent color)
2. **Have compatible value ranges** - their valid values overlap or are from the same domain

Examples of valid groupings:
- **color**: If baseColour has values [red, blue, green] and colour1 has values [red, blue, green, yellow], they can be grouped because they represent the same concept AND share common color values
- **category**: If category has values [electronics, books, clothing] and product_type has values [electronics, books, clothing, sports], they can be grouped
- **price**: If price has range [10, 1000] and discounted_price has range [5, 800], they can be grouped (both are numerical prices)

Examples of INVALID groupings (do NOT group these):
- **color + size**: Even if related to a product, these represent different attributes (color vs dimension) and should be separate groups
- **category + color**: These have different semantic meanings and different value ranges
- Fields with completely incompatible value ranges (e.g., one has [red, blue], another has [small, medium, large])

Grouping Guidelines:
- Create semantic groups based on what the fields represent
- Within each semantic group, verify that value ranges are compatible (overlap or same domain)
- Use descriptive group names that reflect the semantic meaning
- If fields share semantic meaning but have incompatible value ranges, keep them in separate groups or note them individually
- A field should appear in only one group

=== Response Format ===

Return a JSON object with the following structure:
{{
  "value": {{
    "semantic_group_name_1": ["field1", "field2", ...],
    "semantic_group_name_2": ["field3", "field4", ...],
    ...
  }}
}}

IMPORTANT:
- The returned field names must EXACTLY match the column names in the dataset schema
- The fields (together with its available values) should be clearly relevant to the query description.
- The available values of each fields CANNOT be ambiguous (eg, be an ambiguous acronym or have multiple meanings) to be considered compatible.
- Include ALL relevant fields - it's better to be inclusive than exclusive
- Use meaningful semantic group names that describe what the fields represent
- Only group fields that have BOTH semantic similarity AND compatible value ranges
- A field can appear in only one semantic group
- If no fields are clearly relevant, return an empty dictionary {{}}

Output valid JSON only. Do not include explanations or comments.
"""


GENERATE_UCQ_PROMPT = """
You are an expert in query optimization and data filtering.

Your task is to generate a Union of Conjunctive Queries (UCQ) that can safely filter data to find query-relevant records.

=== Query Description ===
{query_desc}

=== Relevant Fields (Grouped Semantically with Value Range Compatibility) ===
The following fields have been identified as relevant to the query, grouped by semantic meaning AND value range compatibility:
{relevant_fields}

**Note**: Fields within each semantic group have been verified to have compatible value ranges (overlapping values or same domain). This means they can safely be used interchangeably or with OR logic.

=== Dataset Schema ===
The dataset contains the following columns with their value distributions:
{schema_info}

=== Using Semantic Groupings ===

Fields within the same semantic group represent related attributes with compatible value ranges. When generating predicates:

1. **For fields within a semantic group:**
   - These fields represent the same concept AND have compatible value ranges (e.g., baseColour, colour1, colour2 all represent color with values like [red, blue, green])
   - Use OR logic between fields in the same semantic group: `(baseColour == 'red' OR colour1 == 'red' OR colour2 == 'red')`
   - Since value ranges are compatible, you can also use multi-value predicates: `color IN ['red', 'blue', 'green']`
   - This ensures you don't miss relevant data regardless of which specific field was populated

2. **For fields across different semantic groups:**
   - These represent independent concepts with different value ranges
   - Use AND logic between fields from different semantic groups: `(price >= 100 AND category == 'electronics')`

3. **Value Range Considerations:**
   - Check the schema info to understand the actual value ranges of fields
   - For categorical fields: verify the specific values available
   - For numerical fields: consider the min/max ranges when setting thresholds
   - Fields grouped together have compatible ranges, but still verify individual values against the schema

4. **When unsure:**
   - Include all potentially relevant fields from a semantic group with OR logic
   - It's safer to be inclusive and filter later than to miss relevant data upfront

=== UCQ Structure ===

You should generate a UCQ represented as a list of conjunctive predicate groups:
- Each inner list represents a conjunctive term (all predicates in the group are combined with AND)
- The outer list combines these conjunctive terms with OR logic
- This allows expressing complex filtering conditions like: (A AND B) OR (C AND D)

For multi-value predicates in equality operators (== or !=), the value can be a list to represent disjunctive conditions within a single field.

=== IMPORTANT SAFETY PRINCIPLES ===

1. MISSING POSITIVE SAMPLES IS UNACCEPTABLE
   - It is FAR better to include some false positives than to miss ANY true positives
   - When in doubt, DO NOT add a filter
   - An empty UCQ is always acceptable

2. ONLY GENERATE PREDICATES WITH VERY HIGH CONFIDENCE
   - Each predicate must be almost certain to be correct
   - Avoid edge cases or ambiguous conditions
   - Prefer simple, unambiguous filters over complex ones

3. CONSERVATIVE FILTERING
   - Use inclusive bounds (>= instead of >, <= instead of <) when possible
   - For categorical data, consider all potentially relevant values using multi-value equality
   - When uncertain between filtering and not filtering, choose not to filter

4. MAKE SURE ONLY THE RELEVANT FIELDS ARE USED
   - Do not include predicates on fields that are not identified as relevant
   - Irrelevant fields can introduce noise and reduce recall

5. Return conjunctive groups that contains at most 4 MOST-important predicates.
   - NEVER return an over-complicated UCQ. Make sure the generated UCQ is as simple as possible.

=== Response Format ===

Return a JSON object with the following structure:
{{
  "value": [  // List of conjunctive groups (OR logic between groups)
    [  // Each group is a list of predicates (AND logic within group)
      {{
        "field": ["field_name"] | ["field1", "field2", ...],  // Single field or merged semantic group
        "op": ">" | ">=" | "<" | "<=" | "==" | "!=",
        "value": [value] | [value1, value2, ...]  // Single value or list of values
      }}
      // ... more predicates in this conjunctive group
    ]
    // ... more conjunctive groups
  ],
  "can_exact_match": true | false  // Whether UCQ exactly matches all desired records
}}

Examples:
- Single field predicate: {{"value": [[{{"field": ["price"], "op": ">=", "value": [100]}}]]}}
- Merged semantic group (color): {{"value": [[{{"field": ["baseColour", "colour1", "colour2"], "op": "==", "value": ["red", "blue"]}}]]}}
- Multi-value inequality: {{"value": [[{{"field": ["status"], "op": "!=", "value": ["deleted", "archived"]}}]]}}
- Mixed predicates: {{"value": [[{{"field": ["baseColour", "colour1"], "op": "==", "value": ["red"]}}, {{"field": ["price"], "op": ">=", "value": [100]}}]]}}
- Empty (no filtering): {{"value": [], "can_exact_match": false}}

**Important Notes:**
- For merged semantic groups (multiple fields in "field"): Use this ONLY when fields have been verified to have compatible value ranges
- For "==" with multiple fields: This creates OR logic between fields (any field can match)
- For "!=" with multiple fields: This creates AND logic between fields (all fields must not match)
- For comparison operators (>, >=, <, <=): Only use single field in "field" list

Output valid JSON only. Do not include explanations or comments.
"""


PROMPTS = {
    "GEN_FEAT_CANDIDATE_PROMPT": GEN_FEAT_CANDIDATE_PROMPT,
    "SUGGEST_FEATURES_PROMPT": SUGGEST_FEATURES_PROMPT,
    "IDENTIFY_RELEVANT_FIELDS_PROMPT": IDENTIFY_RELEVANT_FIELDS_PROMPT,
    "GENERATE_UCQ_PROMPT": GENERATE_UCQ_PROMPT,
}

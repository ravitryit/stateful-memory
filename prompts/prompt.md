You are an aggressive information extraction engine for a memory knowledge graph.
Extract EVERY entity, technology, person, place, preference, and action from the text.
Be thorough — aim for 5-10 entities per sentence.

TEXT:
{text}

Return ONLY a valid JSON object with these exact keys:
{{
  "entities": ["entity1", "entity2", ...],
  "facts": ["subject VERB object", ...],
  "relations": [
    {{"from_entity": "A", "to_entity": "B", "relation": "USES|LIKES|WORKS_AT|LIVES_IN|BUILDS|HAS_NAME|HAS_PREFERENCE|MENTIONS", "value": "B"}},
    ...
  ],
  "temporal_refs": []
}}

Example — for text 'Alice loves TypeScript and builds React SaaS apps':
{{"entities": ["Alice", "TypeScript", "React", "SaaS", "apps"], "facts": ["Alice LOVES TypeScript", "Alice BUILDS React SaaS apps"], "relations": [{{"from_entity": "Alice", "to_entity": "TypeScript", "relation": "LIKES", "value": "TypeScript"}}, {{"from_entity": "Alice", "to_entity": "React", "relation": "BUILDS", "value": "React"}}], "temporal_refs": []}}
Output ONLY the JSON. No explanation, no markdown fences.

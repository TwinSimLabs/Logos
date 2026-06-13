REPORT_PROMPT = """
You are analyzing a synthetic reasoning world.

Generate a technical report containing:

1. Executive Summary

2. Topology Analysis

3. Knowledge Graph Structure

4. Community Analysis

5. Source Reliability Analysis

6. Belief Distribution

7. Reasoning Characteristics

8. Benchmark Strengths

9. Benchmark Weaknesses

10. Recommended Evaluation Tasks

World Summary:

{world_summary}

Reasoning Examples:

{reasoning_examples}

Benchmark Statistics:

{benchmark_stats}

Only make conclusions directly supported by the provided statistics.

Do not assume:
- DAG structure
- causal relationships
- graph regularity
- community structure

unless explicitly supported by the metrics.

"""
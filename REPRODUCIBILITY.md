# Reproducibility

## Environment

Python 3.10+ is recommended.

The original project uses:
- NetworkX
- NumPy
- scikit-learn
- Matplotlib
- sentence-transformers / transformer embeddings
- Google GenAI
- Ollama

Install the declared dependencies in `requirements.txt`.

## Existing artifact analysis

Run:

```bash
python scripts/analyze_graph_structure.py --input got_data.json
python scripts/analyze_graph_structure.py --input improved_data.json
```

The analysis script does not call any external model API.

## Research discipline

For future intervention experiments, pin:
- model identifiers
- prompt template
- number of candidate solutions
- random seed
- embedding model
- similarity threshold
- top-k
- benchmark task set
- objective verifier

Do not report community structure without reporting these configuration values.

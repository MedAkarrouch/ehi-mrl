# Data layout

```text
data/raw/        optional raw downloaded files, ignored by Git
data/processed/  normalized corpus/query/qrels/triples files, ignored by Git
data/embeddings/ cached document/query embeddings, ignored by Git
data/indexes/    FAISS or EHI indexes, ignored by Git
```

Datasets and derived artifacts stay outside version control. The `.gitkeep` files preserve the directory layout only.

# From a raw table corpus to prediction data

> **Task:** Predicting the 'missingness of values' through binary classification

**Input:**: Any raw tabular data.

**Output:**: Prediction table containing embeddings and missingness target $\in$ {1, 0} where 1 is missing else 0.

**Encoders used:** TAPAS-Base, TAPAS-large, MiniLM

**Definitions:**
- *column type*: the target column name (variable name), or the literal text after cleaning headers. For example 'age' or 'age_2' as 'age'
- *annotated type*: a column type might have an annotated type from an existing knowledge base (i.e., schema.org or dbpedia)
- *co-missing columns*: amount of co-missing columns from initial raw dataset
- *sample_id*: during the sampling process, every sample gets an identifier for provenance
- *special_token*: sometimes missing values are no NaNs or NULLs, rather, they are represented as a special token such as 'inapplicable' or 'unknown', these are also safed


### Pipeline steps

1. **Cleaning and preprossessing**: cleans values, headers, outliers, normalization

**inputs**: any parquet file

**output**: parquet file, cleaned

1. **Random sampling from raw data**: samples both missing and non-missing cells from the same column type from a raw input table.

**Inputs**: any parquet file

**Ouputs**: sampling manifest, safed to samples/

This step can be ran on an arbitrary amount of datasets. Includes detection of disguised missing values which are also taken as missing for the pipeline.

*Sampling constraints*
- A row can be sampled at most once
- For every column sampled, an equal amount of missing and non-missing columns should be sampled
- If a column has only present or missing values, no values from the column cannot be sampled

The sample manifest contains:
- sample_id: number assigned to the sample for tractability
- table_id: original dataset/parquet file name
- column_type: identifies the sampled variable/column name
- row_idx: identifies the indices of the sample from the original row for tractability
- y: 1 if the value was missing*, 0 else
- special_token: in case the missing value was a disguised token, the original appearance (e.g., 'inapplicable')
- co_missing_columns: the names of the co-missing columns from the target colum_type

* includes disguised missing data

3. **Encoder input preparation**: prepares input for tokenizer and tokenizes samples for embedding step

This pipeline uses both Table Parsing (TAPAS) architecture and MiniLM. They both require different inputs.

Introduces special tokens to the tokenizers: '[EMPTY]' for missing values and '[?]' to mask the target.

**Inputs**: Sample manifest

**Outputs**:
- TAPAS: sliced row from table
- MiniLM: sentence as a row

Optionally safes tokenized sequences to tokenized/

4. **Build embeddings**

**Inputs**: Tokenized inputs for each model

**Outputs**: Embeddings

5. **Build prediction data**

**Inputs**: Embeddings

**Ouputs**: prediction dataset, safed to prediction_data/

The prediction dataset contains:
- rows of sampled targets, with metadata from sample manifest

### Folder structure

Since ablations are before tokenization time, the prediction datasets are all different:

```
prediction_data/
├── TAPAS-base/
│   ├── default/
│   ├── without_header/
│   ├── without_target/
│   ├── only_target/
│   └── meanpool_target/
├── TAPAS-large/
│   ├── default/
│   ├── without_header/
│   ├── without_target/
│   ├── only_target/
│   └── meanpool_target/
└── MiniLM/
    ├── default/
    ├── without_header/
    ├── without_target/
    ├── only_target/
    └── meanpool_target/
```
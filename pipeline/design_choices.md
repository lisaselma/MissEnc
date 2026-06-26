
### Handling special missing data tokens throughout the pipeline

First, special tokens, meaning missing data encoded as actual values (i.e., disguised missing data) are detected and safed with the use of `00b_special_token_sampler`.

Different special tokens exist, that are not compatible with native missing data handling of numpy/pandas. Examples are "_", "?", "missing", "not applicable", "not specified", "undetermined".

When these are part of the target type, they need to go through the same steps as actual np.NaN's or pd.isna's. For instance, this means that "unknown" should become the target token `[?]`, instead of the string value 'unknown'.

Normalization and table preparation handles this by making sure these special tokens are replaced with actual None. This replacement should not change initial parquet files from /by_semantic_type. In addition, the random sampler counts these as missing values and not other values such as literal strings.

When these special tokens are part of the target/semantic type, these special tokens become the '[?]'-token like any other cell value the pipeline embeds.
When these special tokens are part of other cell-values than the target/semantic type, these tokens should become the '[EMPTY]'-token (native to TAPAS) like any other missing value (np.nan, pd.isna).
### Handling special missing data tokens throughout the pipeline

First, special tokens, meaning missing data encoded as actual values (i.e., disguised missing data) are detected and safed with the use of `00b_special_token_sampler`.

Different special tokens exist, that are not compatible with native missing data handling of numpy/pandas. Examples are "_", "?", "missing", "not applicable", "not specified", "undetermined".

When these are part of the target type, they need to go through the same steps as actual np.NaN's or pd.isna's. For instance, this means that "unknown" should become the target token `[?]`, instead of the string value 'unknown'.

Normalization and table preparation handles this by making sure these special tokens are replaced with actual None. This replacement should not change initial parquet files from /by_semantic_type. In addition, the random sampler counts these as missing values and not other values such as literal strings.

When these special tokens are part of the target/semantic type, these special tokens become the '[?]'-token like any other cell value the pipeline embeds.
When these special tokens are part of other cell-values than the target/semantic type, these tokens should become the '[EMPTY]'-token (native to TAPAS) like any other missing value (np.nan, pd.isna).

The TapasTokenizer has the `empty_token` parameter, characterizing the "[EMPTY]" token. At the same time, transformers/tokeniation_utils.py PretrainedTokenizer.tokenize() explicitly skips empty vals. When "" is passed, the function returns [] without ever reaching TAPAS's _tokenize where the [EMPTY]-token lives. As such, this check should be overcome. The fix for this is to replace all the missing values to the "n/a" string. Empty cell values for TapasTokenizer include "", “n/a”, “nan” and ”?“.

### Sampling random cells for prediction table construction

I sample cells given a target column, with an equal amount of missing cells and non-missing cells. Note that special missing data tokens despites represented as strings (e.g., 'unknown') are also counted as missing data.

parameters:
- n tables to sample from
- n target types per sampled table
- n cells per sampled target type

The TapasTokenizer has the `empty_token` parameter, characterizing the "[EMPTY]" token. At the same time, transformers/tokeniation_utils.py PretrainedTokenizer.tokenize() explicitly skips empty vals. When "" is passed, the function returns [] without ever reaching TAPAS's _tokenize where the [EMPTY]-token lives. As such, this check should be overcome. The fix for this is to replace all the missing values to the "n/a" string. Empty cell values for TapasTokenizer include "", “n/a”, “nan” and ”?“.

### Sampling random cells for prediction table construction

I sample cells given a target column, with an equal amount of missing cells and non-missing cells. Note that special missing data tokens despites represented as strings (e.g., 'unknown') are also counted as missing data.

parameters:
- n tables to sample from
- n target types per sampled table
- n cells per sampled target type

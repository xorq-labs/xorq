How to run the examples
=======================

1. `just download-data` to download the testing data
2. `just up postgres` to launch the postgres instance
3. `uv sync --extra  examples --extra postgres` to install the proper dependencies
3. `pip install .[examples] .[postgres]` to install proper dependencies with pip
4. `brew install libomp`

Then:

```bash
python local_cache.py
```




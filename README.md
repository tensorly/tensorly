# TensorLy

Tensor Learning in Python.


## How to install
 
### Easy/fast option: install with pip

Simply run
```bash
pip install -U tensorly
```
That's it!

### Option 2: clone the repository and install

Clone the repository and cd there:
```bash
git clone https://github.com/tensorly/tensorly
cd tensorly
```

Then install the package (here in editable mode with `-e` or equivalently `--editable`):
```bash
pip install -e .
```

Alternatively, you can pip install from the git repository:
```bash
pip install git+https://github.com/tensorly/tensorly
```

## Running the tests
Testing is an essential part of this package and all functions come with uni-tests.

You can run all the tests using the `nose` package:

```bash
nosetests -v --exe --doctest-tests tensorly
```

1. Acquire Python interpreter version 3.8 from
   [Python.org](https://www.python.org) or with package manager.

   - [Downloads section](https://www.python.org/downloads/)

   - [Last Executable Windows Release](https://www.python.org/downloads/release/python-3810/)

!!! note

    Remember to add Python to PATH.

2. Install tox with pip

   ```
   python -m pip install tox
   ```

3. Create virtual environment with tox:

   ```
   python -m tox -e devenv
   ```

4. Activate environment:

   - Windows: `".tox/devenv/Scripts/activate"`

   - Linux: `source ".tox/devenv/bin/activate"`

   - for more see [this](https://docs.python.org/3/tutorial/venv.html)

!!! info

    If you are using Powershell you may encounter [this problem](https://stackoverflow.com/questions/4037939/powershell-says-execution-of-scripts-is-disabled-on-this-system).

# Tox basics

> A virtual environment is a Python environment such that the Python
> interpreter, libraries and scripts installed into it are isolated from those
> installed in other virtual environments, and (by default) any libraries
> installed in a “system” Python, i.e., one which is installed as part of your
> operating system.

## What is tox?

[`tox`](https://tox.wiki/en/latest/index.html) is a generic virtualenv
management and test command line tool you can use for:

- checking that your package installs correctly with different Python versions
  and interpreters

- running your tests in each of the environments, configuring your test tool of
  choice

- acting as a frontend to Continuous Integration servers, greatly reducing
  boilerplate and merging CI and shell-based testing.

## Handy Links

- [To read more about tox, visit it's documentation.](https://tox.wiki/en/latest/index.html)
- [`tox` global settings](https://tox.wiki/en/latest/config.html#tox-global-settings)
- [`tox` environments configuration](https://tox.wiki/en/latest/config.html#tox-environments)
- [`tox` substitutions](https://tox.wiki/en/latest/config.html#substitutions)
- [Generating environments, conditional settings](https://tox.wiki/en/latest/config.html#generating-environments-conditional-settings)
- [Environmental variables](https://tox.wiki/en/latest/config.html#environment-variables)
- [Full tox cli documentation](https://tox.wiki/en/latest/config.html#cli)
- [`tox` examples](https://tox.wiki/en/latest/examples.html)

---

## Our perspective

Every complex Python project requires multiple tools for development and
deployment. Those are mostly related to test suite running, checking quality of
code, developing and building documentation and building distribution packages.
Usually those are tedious tasks and they are at the top of the lists of tasks
to automate.

**Here comes tox**

`tox` can be used as a reliable replacement for manually written scripts. It's
designed to run predefined series of command in automatically generated Python
virtual environment. It was designed for Python ecosystem and is widely used
along Python projects. It is compatible with other Python tools out of the box.
All the configuration is contained in `tox.ini` file and is completely static.

---

## Basic usage

To invoke single environment with `tox` you have to memorize one simple
command:

```shell
tox -e envname
```

Where `envname` is replaced 1:1 with name of any environments listed below.

!!! Info

    You can also use `tox` command without any arguments to run checks for all
    supported python versions. Be aware that it is really time consuming.

---

## List of all configured environments

Simplicity of creating `tox` managed environments allows us to create highly
specialized environments with minimal boilerplate.

### `devenv`

Stands for development environment (important when using IDE like Visual Studio
Code or PyCharm). When selecting interpreter for your IDE, `devenv` is a right
one to pick.

This environment is meant to contain all tools important for continuos
development including linters, formatters, building tools, packaging tools and
everything else listed in `requirements-dev.txt` It is really heavy and
expensive to create because of complexity of installation. Every call of
`tox -e devenv` will completely recreate the environment.

!!! Danger

    Running tox -e devenv completely reinstalls environment - it's time consuming.

It is designed in such way many due to the fact that during development there
is no need to recreate it until something brakes, and then it's handy to
simplify reinstallation how much possible.

!!! Hint

    Running this environment will install pre-commit.

To select Python from `devenv` as interpreter in Visual Studio Code, use
`Ctrl + Shift + P` and type `Python: Select Interpreter`, then hit `Enter`,
select `Enter interpreter path`, pick `Find` and navigate to `python.exe` in
`.tox/devenv/bin` (unix) or `.tox/devev/scripts` (windows).

This environment is rather bullet proof in comparison to other non-utility
environments (mainly test runners). It should just install on demand, and every
failure should be considered and fixed permanently.

List of included dependencies:

`requirement.txt`

```ini
{% include 'requirements.txt' %}
```

`requirement-test.txt`

```ini
{% include 'requirements-test.txt' %}
```

`requirement-dev.txt`

```ini
{% include 'requirements-dev.txt' %}
```

---

### `check`

Runs formatters and code quality checkers over your workspace.

This environment is lightweight compared to devenv because it installs
dependencies once and completely skips installing a package from this
repository as it does not need it. The operations performed by this environment
are performed in place.

!!! Info

    This environment is lightweight, running tox -e check often is fine.

Similarly to devenv it is bullet proof in comparison to other non-utility
environments (mainly test runners). It should just install on demand, and every
failure should be considered and fixed permanently.

---

### `pyXY`

!!! Warning

    pyXY - test runner envs - they require special care and you are responsible for their well being.

Executes full
[test suite](https://en.wikipedia.org/wiki/Test_suite#:~:text=In%20software%20development%2C%20a%20test,some%20specified%20set%20of%20behaviours.){:target="\_blank"}
with corresponding Python interpreter version, denoted by XX numbers. All
available ones are:

- py37
- py38
- py39
- py310

List of included dependencies:

`requirement.txt`

```ini
{% include 'requirements.txt' %}
```

`requirement-test.txt`

```ini
{% include 'requirements-test.txt' %}
```

---

### `mypy`

Runs mypy over Python codebase to perform static type analysis.

---

### `docs`

Builds documentation with mkdocs, all generated files are saved to `site/`
folder.

---

### `build-all`

Builds package distribution
[wheels](https://realpython.com/python-wheels/#what-is-a-python-wheel){:target="\_blank"}
for corresponding Python version or all versions.

- build-all
- build-py37
- build-py38
- build-py39
- build-py310

Environments with **build** prefix are responsible for building release
packages for corresponding python versions (`build-py37` builds for Python 3.7
etc.) For each test environment (`py37` etc.) there is a corresponding build
environment. Built packages (wheels) are stored in `dist/` directory.

---

## Name tags list

```
devenv
docs
check
py37
py38
py39
py310
mypy
build-all
build-py37
build-py38
build-py39
build-py310
```

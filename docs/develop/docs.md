# Documentation

## Handy links

- [Markdownguide.org Basic Syntax](https://www.markdownguide.org/basic-syntax/){:target="\_blank"}
- [Markdownguide.org Extended Syntax](https://www.markdownguide.org/extended-syntax/){:target="\_blank"}
- [MkDocs-Material Reference](https://squidfunk.github.io/mkdocs-material/reference/){:target="\_blank"}

## MkDocs

[MkDocs](https://www.mkdocs.org/){:target="\_blank"} is a fast, simple and
downright gorgeous static site generator that's geared towards building project
documentation. Documentation source files are written in Markdown, and
configured with a single YAML configuration file. Start by reading the
introductory tutorial, then check the User Guide for more information.

- [Main webpage](https://www.mkdocs.org/){:target="\_blank"}
- [User guide](https://www.mkdocs.org/user-guide/){:target="\_blank"}

We are also using
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/){:target="\_blank"}
theme for documentation which is a separate package.

- [Main webpage](https://squidfunk.github.io/mkdocs-material/){:target="\_blank"}
- [Reference](https://squidfunk.github.io/mkdocs-material/reference/){:target="\_blank"}

```
Usage: mkdocs [OPTIONS] COMMAND [ARGS]...

  MkDocs - Project documentation with Markdown.

Options:
  -V, --version  Show the version and exit.
  -q, --quiet    Silence warnings
  -v, --verbose  Enable verbose output
  -h, --help     Show this message and exit.

Commands:
  build      Build the MkDocs documentation
  gh-deploy  Deploy your documentation to GitHub Pages
  new        Create a new MkDocs project
  serve      Run the builtin development server
```

---

### Live server

Runs development server, which automatically mirrors changes in source code.
Development server is by default available at
[`http://127.0.0.1:8000/`](http://127.0.0.1:8000/){:target="\_blank"}

```shell
mkdocs serve
```

Full CLI help:

```
Usage: mkdocs serve [OPTIONS]

Run the builtin development server

Options:
-a, --dev-addr <IP:PORT>        IP address and port to serve documentation
                                locally (default: localhost:8000)
--livereload                    Enable the live reloading in the development
                                server (this is the default)
--no-livereload                 Disable the live reloading in the
                                development server.
--dirtyreload                   Enable the live reloading in the development
                                server, but only re-build files that have
                                changed
--watch-theme                   Include the theme in list of files to watch
                                for live reloading. Ignored when live reload
                                is not used.
-f, --config-file FILENAME      Provide a specific MkDocs config
-s, --strict                    Enable strict mode. This will cause MkDocs
                                to abort the build on any warnings.
-t, --theme [material|mkdocs|readthedocs]
                                The theme to use when building your
                                documentation.
--use-directory-urls / --no-directory-urls
                                Use directory URLs when building pages (the
                                default).
-q, --quiet                     Silence warnings
-v, --verbose                   Enable verbose output
-h, --help                      Show this message and exit.
```

---

### Build documentation

Builds documentation, all generated files are by default saved to `site/`
folder.

```shell
mkdocs build
```

Full CLI help:

```
Usage: mkdocs build [OPTIONS]

Build the MkDocs documentation

Options:
-c, --clean / --dirty           Remove old files from the site_dir before
                                building (the default).
-f, --config-file FILENAME      Provide a specific MkDocs config
-s, --strict                    Enable strict mode. This will cause MkDocs
                                to abort the build on any warnings.
-t, --theme [mkdocs|material|readthedocs]
                                The theme to use when building your
                                documentation.
--use-directory-urls / --no-directory-urls
                                Use directory URLs when building pages (the
                                default).
-d, --site-dir PATH             The directory to output the result of the
                                documentation build.
-q, --quiet                     Silence warnings
-v, --verbose                   Enable verbose output
-h, --help                      Show this message and exit.
```

---

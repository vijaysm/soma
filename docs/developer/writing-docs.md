# Writing documentation

The pages under `docs/` are the one source of SOMA's documentation. Sphinx builds them into the
Read the Docs site, and `docs/wiki/build_wiki.py` turns the same pages into the repository's
GitHub wiki. Write and change pages only here: each wiki page starts with a comment naming its
source, and the next update of the wiki overwrites any edit made in the wiki itself.

## Adding a page

1. Write `docs/<section>/<page>.md` in MyST Markdown, starting with one `# Title` heading. The
   title also names the wiki page: `# Ghosts and halo exchange` becomes `Ghosts-and-halo-exchange`,
   with characters other than letters, digits, spaces and hyphens dropped. Two pages may not share
   a title.
2. List it, without `.md`, in a `toctree` of `docs/index.md`. A page in no toctree is a Sphinx
   warning, and has no wiki page.
3. Link to other pages with relative links to their files, `../io/overview.md` or
   `../io/overview.md#parallel-reads`, and to the rest of the repository the same way, for
   example `../../examples/app/main.cpp`. In the wiki these become links to the wiki pages and to
   the files on GitHub.
4. Check both outputs:

   ```bash
   python3 -m pip install -r docs/requirements.txt
   sphinx-build -W -b html docs docs/_build/html
   python3 docs/wiki/build_wiki.py --check
   ```

   ctest runs the second as `docs.wiki`, and `docs.wiki_converter` tests the converter itself.

## What the wiki supports

The converter understands what the pages use, and refuses anything else, so a page never reaches
the wiki half converted:

| In `docs/` | In the wiki |
|---|---|
| `# Title` | the page name (the heading is dropped) |
| links to pages, `page.md#anchor` | links to wiki pages; the anchor must name a heading of that page |
| links to other repository files and directories | links to them on GitHub (`main`) |
| ```` ```{toctree} ```` with `:caption:` | a list of links under the caption; those of `index.md` also make `_Sidebar.md` |
| ```` ```{note} ```` and ```` ```{warning} ```` | a block quote starting **Note:** or **Warning:** |
| ```` ```{include} ```` with `:start-line:` | the lines of the included file |
| `$...$` and `$$ ... $$` | left as they are; GitHub renders them |
| code blocks and inline code | left as they are |

Any other directive (`figure`, `math`, `eval-rst`, ...) or role (``{ref}`...` ``, ``{doc}`...` ``)
is an error. Before using one, teach the converter to write its GitHub form, and add a case to
`tests/docs/test_build_wiki.py`.

Anchors are the heading's text in lower case, with characters other than letters, digits,
underscores, hyphens and spaces dropped and spaces made hyphens; MyST and GitHub agree on this,
so `## An application outside SOMA's tree` is `#an-application-outside-somas-tree` in both. A
second heading with the same text gets `-1`, a third `-2`. Only headings of levels 1 to 3 get
anchors in the Sphinx site (`myst_heading_anchors` in `docs/conf.py`).

Files under `docs/` that git ignores are not pages: Sphinx skips them, so private notes can live
there without breaking a local build.

## Publishing

Read the Docs builds the site from `.readthedocs.yaml` on every push. The workflow
`.github/workflows/wiki.yml` builds the site and converts the wiki on pull requests that touch
`docs/`, and on a push to `main` also writes the wiki pages and pushes them to the wiki. The wiki
must exist first: enable it in the repository's settings and save any first page in its editor.
The workflow pushes with its `GITHUB_TOKEN`; if the repository keeps that token read-only, store a
token that may write the repository as the secret `WIKI_TOKEN`.

To update the wiki by hand instead:

```bash
git clone git@github.com:vijaysm/soma.wiki.git ../soma.wiki
python3 docs/wiki/build_wiki.py --out ../soma.wiki
cd ../soma.wiki
git add -A
git commit -m "Update from soma"
git push
```

`--out` rewrites every generated page and deletes the generated pages whose source is gone; pages
written in the wiki by hand, which lack the generated comment, are left alone.

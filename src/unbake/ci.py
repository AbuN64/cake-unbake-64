"""The generated CI of a game repository: one entry point (ci/run) and the thin GitHub and GitLab files that call it.
Everything is a pure function of the packaged rules, the project's versions and the release, so a fresh setup writes
the bytes that are committed."""

from __future__ import annotations

from collections.abc import Sequence
from string import Template

import unbake
from unbake import config as configuration
from unbake.contracts import Json

RUN = "ci/run"
GITHUB = ".github/workflows/ci.yml"
BOARDS = ".github/workflows/boards.yml"
GITLAB = ".gitlab-ci.yml"


def wheel(version: str | None = None) -> str:
    v = unbake.__version__ if version is None else version
    return f"https://github.com/AbuN64/cake-unbake-64/releases/download/v{v}/unbake-{v}-py3-none-any.whl"


def _render(name: str, **values: object) -> bytes:
    return Template(configuration.template(name)).substitute(values).encode()


def _url(board: Json) -> str:
    return str(board["url"]).format(repo="$REPO", version="$v", commit="$COMMIT")


def _shell(board: Json) -> str:
    """The case arm of one board in ci/run."""
    name = board["id"]
    if board["kind"] == "artifact":  # the workflow uploads the artifacts: the arm only proves the reports are there
        body = ['        for v in $VERSIONS; do test -s "versions/$v/report.json"; done']
    else:
        secret = board["secret"]
        body = [
            f'        if [ -z "${{{secret}:-}}" ]; then',
            f'            echo "{name}: the {secret} secret is not set, nothing is published"',
            "            exit 0",
            "        fi",
            "        for v in $VERSIONS; do",
            '            gzip -c "versions/$v/report.json" |',
            f'                curl -fsS -X POST -H "Authorization: Bearer ${secret}" -H "Content-Encoding: gzip" \\',
            f'                --data-binary @- "{_url(board)}"',
            "            echo",
            "        done",
        ]
    return "\n".join([f"    {name})", *body, "        ;;"])


_MONTH = [
    "      - id: month  # artifacts expire: a board that reads them is fed again each month",
    '        run: echo "month=$(date +%Y-%m)" >> "$GITHUB_OUTPUT"',
]


def _github_job(board: Json, pins: Json, versions: Sequence[str]) -> str:
    name, artifact = board["id"], board["kind"] == "artifact"
    month = "-${{ steps.month.outputs.month }}" if artifact else ""
    key = f"board-{name}-${{{{ hashFiles('progress.json') }}}}{month}"
    fresh = f"steps.seen.outputs.cache-hit != 'true' && hashFiles('.boards/{name}') != ''"
    env = ["          REPO: ${{ github.repository }}", "          COMMIT: ${{ github.sha }}"]
    if not artifact:
        env.append(f"          {board['secret']}: ${{{{ secrets.{board['secret']} }}}}")
    lines = [
        f"  {name}:",
        "    runs-on: ubuntu-latest",
        "    steps:",
        f"      - uses: {pins['checkout']}",
        *(_MONTH if artifact else []),
        "      - id: seen",
        "        if: ${{ !inputs.force }}",
        f"        uses: {pins['cache_restore']}",
        "        with:",
        f"          path: .boards/{name}",
        f"          key: {key}",
        "      - env:",
        *env,
        f"        run: sh ci/run board {name}",
    ]
    if artifact:
        for v in versions:
            lines += [
                f"      - if: {fresh}",
                f"        uses: {pins['upload']}",
                "        with:",
                f"          name: {board['artifact'].format(version=v)}",
                f"          path: versions/{v}/report.json",
                "          retention-days: 90",
            ]
    lines += [
        f"      - if: {fresh}",
        f"        uses: {pins['cache_save']}",
        "        with:",
        f"          path: .boards/{name}",
        f"          key: {key}",
    ]
    return "\n".join(lines)


def _gitlab_job(board: Json, gitlab: Json) -> str:
    """Upload boards only: an artifact board reads GitHub Actions artifacts."""
    name = board["id"]
    return "\n".join(
        [
            "",
            f"board-{name}:",
            f"  image: {gitlab['audit_image']}",
            "  rules:",
            '    - if: $CI_PIPELINE_SOURCE == "schedule" || $CI_PIPELINE_SOURCE == "web"',
            "  cache:",
            "    key:",
            f"      prefix: board-{name}",
            "      files:",
            "        - progress.json",
            "    paths:",
            f"      - .boards/{name}",
            "  variables:",
            "    REPO: $CI_PROJECT_PATH",
            "    COMMIT: $CI_COMMIT_SHA",
            "  before_script:",
            "    - apt-get update",
            f"    - apt-get install -y --no-install-recommends {' '.join(gitlab['board_packages'])}",
            "  script:",
            f"    - sh ci/run board {name}",
            "",
        ]
    )


def files(versions: Sequence[str]) -> dict[str, bytes]:
    rules, boards = configuration.load_resource("repo.toml"), configuration.load_resource("boards.toml")["board"]
    pins = {k: f"{r['action']}@{r['sha']} # {r['tag']}" for k, r in rules["github"].items()}
    gitlab = rules["gitlab"]
    return {
        RUN: _render("run.in", wheel=wheel(), versions=" ".join(versions), boards="\n".join(map(_shell, boards))),
        GITHUB: _render("ci.yml.in", checkout=pins["checkout"], python=pins["python"]),
        BOARDS: _render("boards.yml.in", jobs="\n".join(_github_job(b, pins, versions) for b in boards) + "\n"),
        GITLAB: _render(
            "gitlab-ci.yml.in",
            image=gitlab["image"],
            audit_image=gitlab["audit_image"],
            packages=" ".join(gitlab["packages"]),
            boards="".join(_gitlab_job(b, gitlab) for b in boards if b["kind"] == "upload"),
        ),
    }

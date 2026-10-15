"""The generated CI: one entry point, thin GitHub and GitLab callers, a daily boards workflow with a job per board."""

import yaml

import unbake
from unbake import ci
from unbake import config as configuration

VERSIONS = ("us", "eu-x")


def _files():
    return {path: data.decode() for path, data in ci.files(VERSIONS).items()}


def test_rendering_is_a_pure_function_of_the_rules_versions_and_release():
    assert ci.files(VERSIONS) == ci.files(VERSIONS)
    run = _files()[ci.RUN]
    assert f"releases/download/v{unbake.__version__}/unbake-{unbake.__version__}-py3-none-any.whl" in run
    assert 'VERSIONS="us eu-x"' in run


def test_every_job_calls_the_one_entry_point():
    files = _files()
    github = yaml.safe_load(files[ci.GITHUB])
    assert set(github[True]) == {"pull_request", "push", "workflow_dispatch"}  # yaml reads the key `on` as True
    runs = [step["run"] for job in github["jobs"].values() for step in job["steps"] if "run" in step]
    assert runs == ["sh ci/run check", "sh ci/run audit"]
    gitlab = yaml.safe_load(files[ci.GITLAB])
    assert {name: job["script"][-1].split("sh ci/run ")[1] for name, job in gitlab.items()} == {
        "check": "check",
        "audit": "audit",
        "board-bytematch": "board bytematch",
    }
    assert [path for path, text in files.items() if "unbake audit" in text] == [ci.RUN]


def test_boards_run_daily_one_job_per_board_and_never_on_push():
    boards = configuration.load_resource("boards.toml")["board"]
    workflow = yaml.safe_load(_files()[ci.BOARDS])
    assert set(workflow[True]) == {"schedule", "workflow_dispatch"}
    assert len(workflow[True]["schedule"]) == 1 and workflow[True]["schedule"][0]["cron"].endswith("* * *")
    assert set(workflow["jobs"]) == {b["id"] for b in boards}
    for job in workflow["jobs"].values():
        assert "needs" not in job  # one board failing never blocks another


def test_a_board_publishes_only_after_progress_changed():
    workflow = yaml.safe_load(_files()[ci.BOARDS])
    for name, job in workflow["jobs"].items():
        steps = {
            s.get("id") or s.get("run") or s["uses"].split("@")[0] + s.get("with", {}).get("name", ""): s
            for s in job["steps"]
        }
        seen = steps["seen"]
        assert seen["uses"].startswith("actions/cache/restore@") and seen["if"] == "${{ !inputs.force }}"
        assert seen["with"]["key"].startswith(f"board-{name}-${{{{ hashFiles('progress.json') }}}}")
        save = steps["actions/cache/save"]
        assert save["with"] == seen["with"] and "steps.seen.outputs.cache-hit != 'true'" in save["if"]
        assert f"hashFiles('.boards/{name}') != ''" in save["if"]  # only a board that published records it
        assert steps[f"sh ci/run board {name}"]["env"]["REPO"] == "${{ github.repository }}"


def test_an_artifact_board_uploads_each_version_and_refreshes_monthly():
    job = yaml.safe_load(_files()[ci.BOARDS])["jobs"]["recomp"]
    uploads = [s["with"] for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert uploads == [
        {"name": f"{v}_report", "path": f"versions/{v}/report.json", "retention-days": 90} for v in VERSIONS
    ]
    assert job["steps"][1]["id"] == "month" and job["steps"][2]["with"]["key"].endswith(
        "-${{ steps.month.outputs.month }}"
    )


def test_an_upload_board_posts_every_report_with_its_secret():
    run = _files()[ci.RUN]
    arm = run.split("    bytematch)\n")[1].split(";;")[0]
    assert 'if [ -z "${BYTEMATCH_TOKEN:-}" ]' in arm and "exit 0" in arm  # no secret yet: nothing, and no record
    assert '"https://bytematch.dev/api/v1/projects/$REPO/reports?version=$v&commit=$COMMIT"' in arm
    assert 'gzip -c "versions/$v/report.json"' in arm and "curl -fsS" in arm
    job = yaml.safe_load(_files()[ci.BOARDS])["jobs"]["bytematch"]
    (step,) = [s for s in job["steps"] if s.get("run") == "sh ci/run board bytematch"]
    assert step["env"]["BYTEMATCH_TOKEN"] == "${{ secrets.BYTEMATCH_TOKEN }}"


def test_the_entry_point_records_a_board_only_after_it_published():
    run = _files()[ci.RUN]
    board = run.split("board)\n")[1]
    assert (
        board.index('if [ -f ".boards/$name" ]')
        < board.index('    case "$name" in')
        < board.index(': >".boards/$name"')
    )
    assert "set -eu" in run  # a failed upload ends the script before the record


def test_audit_base_falls_back_for_a_new_branch_and_a_shallow_base():
    run = _files()[ci.RUN]
    assert "*0000000000000000000000000000000000000000) b=4b825dc642cb6eb9a060e54bf8d69288fbee4904" in run
    assert 'git cat-file -e "$b^{tree}" 2>/dev/null || b=HEAD' in run


def test_pins_are_full_digests():
    text = "".join(_files().values())
    for row in configuration.load_resource("repo.toml")["github"].values():
        assert f"{row['action']}@{row['sha']} # {row['tag']}" in text
    for line in text.splitlines():
        if "uses:" in line:
            assert "@" in line and len(line.split("@")[1].split()[0]) == 40


def test_no_game_names():
    text = "".join(_files().values()).lower() + str(configuration.load_resource("boards.toml")).lower()
    assert not [word for word in ("ragewars", "battletanx", "turok") if word in text]

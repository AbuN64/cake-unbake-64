# Broker resource domains

The host's `[resources].domain` selects either `"standalone"` (one command alone,
using its own pool and configured limits without cgroups or sockets) or an absolute
TOML manifest path. No mode is selected by default, and project overrides cannot
set `domain`. In broker mode, host `cores` and `memory_total_bytes` request one
whole-command grant. Pool size is bounded by cores, workers and memory; make uses
cores, and native children inherit the granted cgroup.

The manifest contains exactly these values:

| Key | Meaning |
| --- | --- |
| `root` | Common cgroup v2 domain containing control and jobs |
| `control` | Bounded subtree for waiting command parents and the broker |
| `jobs` | Empty delegated jobs directory inside the broker service's original cgroup |
| `cores`, `memory_bytes` | Aggregate capacity matching root's kernel limits |
| `control_cores`, `control_memory_bytes` | Reserved control capacity matching its kernel limits |
| `bypass_limit` | Maximum smaller requests allowed to pass an older queued request |

Paths are absolute; control and jobs are disjoint descendants of root. Numbers
are positive integers and control reservations are smaller than aggregate limits.
Swap must be disabled for root and control. The broker also subtracts observed
initial memory charges, since moving a PID does not migrate existing charges.
Oversized requests and manifest disagreement refuse before project work.

The deployment owner provisions the limits and a delegated systemd user service
with `Delegate=cpu memory` and `KillMode=control-group`. Its main PID must be tracked
and all job descendants killed on every service exit, including SIGKILL. Start it
in the delegated parent of jobs with:

```sh
/absolute/pinned/unbake resources /absolute/resource-domain.toml
```

The broker moves into a control leaf, enables controllers and creates jobs. Retained
job directories refuse startup and require deployment recovery. One abstract UNIX
socket identifies the physical root; all participants use the same host/network
namespace and manifest. Launch project commands inside control before children or
a forkserver exist, using the immutable tool/dependency path for that invocation.

CPU and memory are granted together before the project writer lock and project
preparation. Bounded overtaking prevents older requests from starving. A closed
socket does not release live work: pidfd death triggers subgroup termination,
and capacity returns only when descendants drain. Broker stdout emits JSONL grants
and releases; command effort includes queue time and cgroup measurements. This
admission does not change project writer lock scope.

Kernel enforcement requires a real deployment proof with overlapping public
commands, serial-reference outputs, aggregate CPU/memory measurements and failed
or cancelled subtree drainage. Mock cgroup tests prove accounting, not enforcement.

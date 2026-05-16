# Control Upstream

`control/` tracks the standalone excavator control library.

| Field | Value |
|-------|-------|
| Repository | https://github.com/Asukayoo/excavator.git |
| Branch | `main` |
| Synced commit | `cc286a1` |
| Monorepo path | `control/` |

## Sync from upstream

```bash
# In excavator_control (or any clone of Asukayoo/excavator)
git fetch origin
git log --oneline <last_synced_commit>..origin/main

git format-patch -<N> origin/main -o /tmp/control-patches

# In Excavator_real_stack
git am --directory=control /tmp/control-patches/*.patch
```

After each sync, update `Synced commit` in this file and the note in the root `README.md`.

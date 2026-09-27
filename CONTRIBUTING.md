# Contributing to AegisEdge

Thanks for looking. This is a small project with strong opinions about
evidence, so most of what follows is about how a change earns its way in
rather than about formatting.

## The short version

```bash
git clone https://github.com/akshatinnovate-png/AegisEdge.git
cd AegisEdge
pip install -r backend/requirements-dev.txt

cd backend
python3 -m pytest -q                    # 347 tests
python3 scripts/audit_determinism.py    # the determinism lint
python3 scripts/audit_claims.py         # the docs' numbers, checked
```

All three run in CI on every push. A change that breaks any of them will not
merge, and the failure messages are written to tell you what to do.

## What a good change looks like

**A defect fix comes with the test that would have caught it.** Not a test
that passes afterwards — a test that fails before. Several of this project's
bugs survived a long time because the test written alongside them could not
have failed; there is a whole section about it in the README.

**A measured claim comes with the measurement.** If a change makes something
faster, say by how much, on what, and include the harness. `backend/scripts/`
has several to copy from. A number without a way to reproduce it is a
decoration.

**A limit gets written down.** Every part of this system that does not work,
or works only under conditions, says so in the README. That is not modesty;
it is the difference between a reader who can plan around the system and one
who finds out later.

## What will get pushed back on

- **A new dependency**, unless it earns itself. `backend/requirements.txt` is
  short on purpose and `tests/test_packaging.py` fails if an import is not
  declared.
- **A claim in the docs with no artefact behind it.** `scripts/audit_claims.py`
  checks the ones that are declared; the convention is to declare yours too.
- **Anything in `aegis/sync/` or `aegis/core/` that reads the clock or the
  random number generator directly.** Use `aegis.core.determinism`. The lint
  will tell you, and the reason is that a seed has to reproduce exactly or the
  simulator is worthless.
- **Skipping, disabling or quarantining a failing test** to get a green run.

## Running the simulator

The deterministic simulator is where most of the interesting bugs came from.
It is worth knowing:

```bash
python3 scripts/simulate.py --seeds 200                       # ~2 min
python3 scripts/simulate.py --replay 5                        # one seed, in full
python3 scripts/simulate.py --seeds 200 --unsigned --no-shrink \
        --stop-after 200                                      # the control
```

If you add an invariant, add the control too: run it with the defence removed
and check that it fails. An invariant that cannot fail is worse than none,
because the counter it feeds says everything is fine. Two of this project's
seven spent three thousand executions in exactly that state.

## Style

Match the file you are editing. Comments explain *why*, especially where the
obvious approach was tried and did not work — a lot of this codebase's
comments are records of measurements that contradicted an intuition, and they
are the most useful thing in it.

## Reporting something

Open an issue with what you did, what happened, and what you expected. If it
is a simulator failure, the seed is enough — every execution is a pure
function of its seed, so a seed reproduces it exactly, on any machine.

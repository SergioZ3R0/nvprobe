## Description
Please include a summary of the change and the motivation. If it fixes an open bug, please link to the issue.

Fixes # (issue number)

## Type of change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (e.g., new benchmark, new chart in the HTML report)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?
Please describe the tests that you ran to verify your changes.
- [ ] Tested locally (`nvprobe run --local`)
- [ ] Tested via Slurm submission (`nvprobe slurm submit`) if applicable
- [ ] Verified that the HTML report generates correctly (`nvprobe report --open`)
- [ ] Verified MLPerf inference still runs (if touching the MLPerf module)

## Checklist:
- [ ] My code follows the existing style of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in complex CUDA or Slurm interactions
- [ ] I have updated the `README.md` if new commands or configurations were added

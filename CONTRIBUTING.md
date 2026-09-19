# Contributing to nvprobe

We welcome contributions to nvprobe. This document outlines the standard process for contributing to the project.

## Reporting Issues

Before opening a new issue, please check the existing open issues to avoid duplicates.

When reporting a bug or requesting a feature, use the provided GitHub issue templates. Ensure you include relevant details about your hardware and software environment, such as:
- OS distribution and version
- GPU model and count
- CUDA version (`nvprobe env`)
- Python version
- Execution mode (Local or Slurm)

## Local Development Setup

To set up a local development environment, follow these steps:

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nvprobe.git
   cd nvprobe
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install the package in editable mode:
   ```bash
   pip install -e .
   ```
   Changes made to the source code will now be applied immediately when running the `nvprobe` command.

## Pull Request Process

- **Branching**: Create a new branch for your changes (`git checkout -b feature/issue-name`).
- **Code Style**: Maintain consistency with the existing codebase.
- **Testing**: Verify your changes do not break existing functionality. Run the following baseline checks before submitting a PR:
  - `nvprobe env`
  - `nvprobe run --local` (or target a specific benchmark module)
  - `nvprobe report --open`
- **Commit Messages**: Write clear, concise, and descriptive commit messages.
- **Submission**: Open a Pull Request against the `main` branch. Fill out the provided Pull Request template completely, linking any relevant issues (e.g., `Fixes #10`).

All Pull Requests will be reviewed by the maintainers before merging.

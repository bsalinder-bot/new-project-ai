# Contributing to SYMBIOTIC PLANETARY INTELLIGENCE (SPI)

## Coding Style
- Use `snake_case` for functions and variables.
- Use `PascalCase` for class names.
- Keep modules small, focused, and easy to test.
- Add type hints for public methods and functions where possible.
- Use descriptive names and avoid abbreviations.

## Error Handling
- Validate external inputs before processing.
- Raise specific exceptions for invalid data.
- Return structured API errors from the Flask endpoints.
- Do not expose internal tracebacks in production API responses.

## Testing
- Every feature must include unit tests.
- Test validation, edge cases, and failure paths.
- Run tests with:
  ```bash
  python -m unittest discover -s tests
  ```

## Git Workflow
- Create feature branches from `main`.
- Keep commits atomic and descriptive.
- Rebase or merge consistently with `main`.
- Run tests locally before pushing changes.

## Review Process
- Use the PR template in `.github/pull_request_template.md`.
- Add reviewers in GitHub using the Reviewers panel.
- If you need a review from external developers, use `REVIEW_REQUEST.md` as a message template.
- Use issue templates in `.github/ISSUE_TEMPLATE/` for bug reports and feature requests.

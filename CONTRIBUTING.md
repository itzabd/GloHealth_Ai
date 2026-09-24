# Contributing to GloHealth AI

Thank you for your interest in contributing to **GloHealth AI**! We welcome community contributions to help improve disease prediction, healthcare accessibility, and epidemiological surveillance.

---

## Code of Conduct

By participating in this project, you agree to maintain a welcoming, respectful, and inclusive environment for all contributors.

---

## How Can I Contribute?

### 1. Reporting Bugs
- Search existing [GitHub Issues](https://github.com/itzabd/GloHealth_Ai/issues) to avoid duplicate reports.
- If not already reported, create a new issue using our **Bug Report** template.
- Include clear reproduction steps, environment details (OS, Python version), and expected vs. actual behavior.

### 2. Suggesting Enhancements
- Open a feature request issue explaining the motivation, use case, and proposed solution.
- For UI/UX changes, mockups or wireframes are appreciated.

### 3. Submitting Pull Requests (PRs)
1. **Fork** the repository and create your branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Set up your local environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env
   ```
3. **Adhere to coding standards**:
   - Follow **PEP 8** style guidelines for Python code.
   - Use meaningful variable and function names.
   - Add docstrings and inline comments for complex algorithms or ML transformations.
   - Ensure templates remain responsive across mobile, tablet, and desktop viewports.
4. **Test your changes**:
   - Verify that python syntax checks pass without errors:
     ```bash
     python -m py_compile app.py geo_analysis.py data_prep.py train_model.py
     ```
   - Run the local application:
     ```bash
     python app.py
     ```
5. **Commit your changes**:
   - Use concise, conventional commit messages (e.g., `feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`).
6. **Open a Pull Request**:
   - Fill out our PR template with a clear explanation of changes made and verification steps performed.

---

## Machine Learning & Data Guidelines

- When proposing changes to ML models or training pipelines, provide validation metrics (F1-score, accuracy, confusion matrix) in your PR description.
- Never commit private patient data or unverified datasets. Ensure all training datasets comply with data privacy policies and open licenses.

---

## Questions or Need Help?

Feel free to open a discussion or issue on GitHub, or reach out to the project maintainer:
- **Maintainer:** Abdullah Hossien ([@itzabd](https://github.com/itzabd))
- **Live Platform:** [glohealth-ai.onrender.com](https://glohealth-ai.onrender.com)


# Project Name
Tomoro Assigment

## Project Setup

Follow the instructions below to set up the project on your local machine.

### 1. Obtain an OpenAI API Key
To use this project, you must have an OpenAI API key. Follow these steps:

1. Go to [OpenAI](https://platform.openai.com/account/api-keys) and log in or sign up.
2. Generate a new API key.
3. Add the API key to your environment variables:

   **On Linux/Mac:**
   ```bash
   export OPENAI_API_KEY='your_api_key_here'
   ```

   **On Windows (Command Prompt):**
   ```cmd
   set OPENAI_API_KEY=your_api_key_here
   ```

   **On Windows (PowerShell):**
   ```powershell
   $env:OPENAI_API_KEY = "your_api_key_here"
   ```

### 2. Install Project Dependencies

To ensure that all necessary dependencies and development tools are available, follow these steps:

1. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

These installations will cover all the necessary packages for running and developing this project.

### 3. Download the Dataset

Ensure that the `train.json` dataset is stored in the `data/` directory of your project. This dataset can be obtained from the [ConvFinQA GitHub repository](https://github.com/czyssrs/ConvFinQA).

1. Navigate to the `data/` directory in your project root:
   ```bash
   mkdir -p data
   cd data
   ```

2. Download or copy the `train.json` file into this directory.

---

### Additional Notes
- Ensure that Python is installed on your system. This project requires **Python 3.7** or higher.
- For virtual environment setup, consider using `venv` or `virtualenv` to keep dependencies isolated:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

## Running the Project
After completing the setup steps, you can run the project using:
```bash
python main.py  # Replace 'main.py' with the entry-point file if different.
```

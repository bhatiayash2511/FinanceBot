
# Project Setup Guide

This guide walks you through the steps to set up and run the project, including creating the environment, running the database script, and launching the application using Streamlit.

## 1. Create Environment Using `requirements.txt`

First, create a virtual environment and install the necessary dependencies using the `requirements.txt` file.

### Steps: 
1. Clone or download the project repository.
2. Open a terminal and navigate to the project directory.
3. Create a virtual environment:
   - **On Windows**:
     ```bash
     python -m venv venv
     ```
   - **On macOS/Linux**:
     ```bash
     python3 -m venv venv
     ```
4. Activate the virtual environment:
   - **On Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **On macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Set Up Environment Variables

Create a `.env` file in the root of your project directory and add the following environment variables:

```env
LANGCHAIN_API_KEY="your_api_key_here"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_PROJECT="your_project_name_here"
KMP_DUPLICATE_LIB_OK=True
```

### Install the Python `python-dotenv` package:
```bash
pip install python-dotenv
```

### Load the environment variables in your Python code:

```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the variables
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_project = os.getenv("LANGCHAIN_PROJECT")
kmp_duplicate_lib_ok = os.getenv("KMP_DUPLICATE_LIB_OK")
```

## 3. Run `db.py` to Get Vector Database

Once the environment is set up, you need to run the `db.py` script to create and populate the vector database.

### Steps:
1. In the terminal, run the `db.py` script:
   ```bash
   python db.py
   ```

   This will generate the vector database required for the application to function.

## 4. Run the Application Using Streamlit

Now that the database is set up, you can run the application using Streamlit.

### Steps:
1. In the terminal, run the following command:
   ```bash
   streamlit run app.py
   ```

2. This will launch the application in your default web browser.

## Troubleshooting

- If you encounter issues with the `requirements.txt`, make sure your virtual environment is activated and that all dependencies are correctly installed.
- Ensure that the necessary files (like `db.py` and `app.py`) are present in the project directory.

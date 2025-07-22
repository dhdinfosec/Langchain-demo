#!/bin/bash

# ==============================================================================
#
# Project Setup Script for LangGraph RAG Agent
#
# This script automates the setup of the project environment. It performs
# the following actions:
# 1. Creates a dedicated project directory if it doesn't exist.
# 2. Creates a Python virtual environment (venv) within that directory.
# 3. Activates the virtual environment.
# 4. Installs all required Python packages from a predefined list.
# 5. Creates or updates a .env file with placeholder API keys for the user to
#    fill in.
#
# The script is designed to be idempotent, meaning it can be run multiple
# times without causing issues. It will detect existing directories and
# installed packages and skip them accordingly.
#
# ==============================================================================


# --- Configuration Variables ---
# The name of the main directory where the project files will be stored.
PROJECT_DIR="langgraph-agent"
# The standard name for the virtual environment subdirectory.
VENV_SUBDIR_NAME="venv"


# --- Path and Environment Setup ---
# Get the absolute path to the directory where this setup.sh script is located.
# This is crucial for reliably constructing other absolute paths, ensuring the
# script works correctly regardless of where it's called from.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Construct the absolute path to the main project directory.
ABS_PROJECT_DIR="$SCRIPT_DIR/$PROJECT_DIR"

# This variable will be populated with the full path to the virtual
# environment's 'activate' script.
ACTIVATE_SCRIPT_PATH=""


# --- Detect or Create Virtual Environment ---
# This block handles two main scenarios:
# 1. The project directory already exists, so we just need to find the venv.
# 2. The project directory doesn't exist, so we need to create it and the venv.

# Case 1: The project directory already exists.
if [ -d "$ABS_PROJECT_DIR" ]; then
    echo "Directory '$ABS_PROJECT_DIR' already exists."

    # Within the existing directory, check for common venv structures.
    if [ -f "$ABS_PROJECT_DIR/bin/activate" ]; then
        echo "Detected existing flat virtual environment structure within '$ABS_PROJECT_DIR'."
        ACTIVATE_SCRIPT_PATH="$ABS_PROJECT_DIR/bin/activate"
    elif [ -f "$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME/bin/activate" ]; then
        echo "Detected existing standard virtual environment structure within '$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME'."
        ACTIVATE_SCRIPT_PATH="$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME/bin/activate"
    else
        # If the directory exists but contains no venv, it's a configuration error.
        echo "ERROR: '$ABS_PROJECT_DIR' exists, but no 'bin/activate' script was found."
        exit 1
    fi
else
    # Case 2: The project directory does NOT exist. Create everything from scratch.
    echo "Creating directory '$ABS_PROJECT_DIR'..."
    mkdir "$ABS_PROJECT_DIR" || { echo "ERROR: Failed to create directory. Exiting."; exit 1; }

    echo "Creating new virtual environment in '$VENV_SUBDIR_NAME'..."
    # Use 'python3 -m venv' to create the virtual environment inside the project directory.
    (cd "$ABS_PROJECT_DIR" && python3 -m venv "$VENV_SUBDIR_NAME") || \
        { echo "ERROR: Failed to create virtual environment. Exiting."; exit 1; }

    # Set the path to the activate script for the newly created venv.
    ACTIVATE_SCRIPT_PATH="$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME/bin/activate"
fi

# IMPORTANT: Change the current working directory to the project root.
cd "$ABS_PROJECT_DIR" || { echo "ERROR: Failed to change directory. Exiting."; exit 1; }


# --- Activate the Virtual Environment ---
# Deactivate any previously active virtual environment in the current shell.
if type deactivate &>/dev/null; then
    deactivate
fi

# Activate our target virtual environment using its absolute path.
echo "Activating virtual environment from '$ACTIVATE_SCRIPT_PATH'..."
if ! source "$ACTIVATE_SCRIPT_PATH"; then
    echo "ERROR: Failed to activate virtual environment. Exiting."
    exit 1
fi


# --- Install Required Python Packages ---
echo "Checking and installing Python packages..."
# Defines the list of Python packages required for the project.
PYTHON_PACKAGES="langgraph langchain_core langchain-google-genai pydantic python-dotenv langsmith jmespath termcolor langchain-community langchain-tavily beautifulsoup4"

# Loop through each package in the list.
for pkg in $PYTHON_PACKAGES; do
    # Use 'python3 -m pip show' to check if the package is already installed.
    if ! python3 -m pip show "$pkg" &>/dev/null; then
        echo "  Installing $pkg..."
        # Use 'python3 -m pip install' to ensure we are using the pip from our
        # activated virtual environment. Added --quiet flag to reduce install noise.
        python3 -m pip install --quiet "$pkg" || { echo "  ERROR: Failed to install $pkg. Exiting."; exit 1; }
    else
        echo "  $pkg is already installed."
    fi
done


# --- Create .env File for API Keys ---
# Use the absolute path to avoid ambiguity.
ABS_ENV_FILE="$ABS_PROJECT_DIR/.env"

echo "Checking for .env file..."
# Check if the .env file does NOT exist.
if [ ! -f "$ABS_ENV_FILE" ]; then
    echo "Creating .env file with placeholder API keys..."
    # Create the file and add placeholder keys. The user must replace these.
    echo "LANGSMITH_API_KEY=\"your-langsmith-key-here\"" > "$ABS_ENV_FILE"
    echo "GOOGLE_API_KEY=\"your-gemini-api-key-here\"" >> "$ABS_ENV_FILE"
    echo "TAVILY_API_KEY=\"your-tavily-api-key-here\"" >> "$ABS_ENV_FILE"
else
    echo ".env file already exists. Skipping creation."
    # Check if TAVILY_API_KEY is missing from an existing file and add it if so.
    if ! grep -q "TAVILY_API_KEY" "$ABS_ENV_FILE"; then
        echo "  Adding TAVILY_API_KEY placeholder to existing .env file..."
        echo "TAVILY_API_KEY=\"your-tavily-api-key-here\"" >> "$ABS_ENV_FILE"
    fi
fi


# --- Final Instructions ---
# Print a confirmation message and next steps for the user.
echo ""
echo "Setup complete!"
echo "1. Remember to edit '$ABS_ENV_FILE' with your actual API keys."
echo "2. Activate your venv in new terminal sessions using: source $ACTIVATE_SCRIPT_PATH"

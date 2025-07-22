#!/bin/bash

# ==============================================================================
#
# Project Setup Script for LangGraph Agent
#
# This script automates the setup of the project environment. It performs
# the following actions:
# 1. Creates a dedicated project directory if it doesn't exist.
# 2. Creates a Python virtual environment (venv) within that directory.
# 3. Activates the virtual environment.
# 4. Installs all required Python packages from a predefined list.
# 5. Creates a .env file with placeholder API keys for the user to fill in.
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
# The name of the environment file for storing API keys.
ENV_FILE="$PROJECT_DIR/.env"


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
    # First, check for a "flat" structure (e.g., project was the venv root).
    if [ -f "$ABS_PROJECT_DIR/bin/activate" ]; then
        echo "Detected existing flat virtual environment structure within '$ABS_PROJECT_DIR'."
        ACTIVATE_SCRIPT_PATH="$ABS_PROJECT_DIR/bin/activate"
    # Next, check for the standard structure where the venv is in a subdirectory.
    elif [ -f "$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME/bin/activate" ]; then
        echo "Detected existing standard virtual environment structure within '$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME'."
        ACTIVATE_SCRIPT_PATH="$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME/bin/activate"
    else
        # If the directory exists but contains no venv, it's a configuration error.
        echo "ERROR: '$ABS_PROJECT_DIR' exists, but no 'bin/activate' script was found (neither flat nor in '$VENV_SUBDIR_NAME/')."
        echo "Please ensure a valid virtual environment exists within '$ABS_PROJECT_DIR'."
        exit 1
    fi
else
    # Case 2: The project directory does NOT exist. Create everything from scratch.
    echo "Creating directory '$ABS_PROJECT_DIR'..."
    mkdir "$ABS_PROJECT_DIR" || { echo "ERROR: Failed to create directory $ABS_PROJECT_DIR. Exiting."; exit 1; }

    echo "Creating new virtual environment in '$VENV_SUBDIR_NAME' inside '$ABS_PROJECT_DIR'..."
    # Use 'python3 -m venv' to create the virtual environment inside the project directory.
    (cd "$ABS_PROJECT_DIR" && python3 -m venv "$VENV_SUBDIR_NAME") || \
        { echo "ERROR: Failed to create virtual environment in '$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME'. Exiting."; exit 1; }

    # Set the path to the activate script for the newly created venv.
    ACTIVATE_SCRIPT_PATH="$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME/bin/activate"
fi

# IMPORTANT: Change the current working directory to the project root.
# This is done *after* all absolute paths have been determined but *before*
# activation. This ensures that subsequent commands like 'pip' install
# packages in the correct context and that the user can easily run commands
# from the project root.
cd "$ABS_PROJECT_DIR" || { echo "ERROR: Failed to change directory to $ABS_PROJECT_DIR. Exiting."; exit 1; }


# --- Activate the Virtual Environment ---
# Deactivate any previously active virtual environment in the current shell.
# This prevents potential conflicts between different Python environments.
if type deactivate &>/dev/null; then
    echo "Deactivating any existing virtual environment..."
    deactivate
fi

# Activate our target virtual environment using its absolute path.
# The 'source' command executes the script in the current shell context,
# modifying the PATH to prioritize the venv's Python and pip installations.
echo "Activating virtual environment from '$ACTIVATE_SCRIPT_PATH'..."
if ! source "$ACTIVATE_SCRIPT_PATH"; then
    echo "ERROR: Failed to activate virtual environment. Exiting."
    exit 1
fi


# --- DEBUGGING OUTPUT (can be safely removed) ---
# This section prints information about the current environment after activation
# to help diagnose any PATH or installation issues.
echo "--- DEBUGGING AFTER ACTIVATION ---"
echo "Current working directory: `pwd`"
echo "Path to Python executable:"
which python
which python3
echo "Path to pip executable:"
which pip
echo "Current PATH variable:"
echo "PATH: $PATH"
echo "--- End Debugging ---"


# --- Install Required Python Packages ---
echo "Checking and installing Python packages..."
# A space-separated list of all Python packages required for the project.
PYTHON_PACKAGES="langgraph langchain_openai langchain_core langchain-google-genai wikipedia pydantic python-dotenv langsmith jmespath termcolor"

# Loop through each package in the list.
for pkg in $PYTHON_PACKAGES; do
    # Use 'python3 -m pip show' to check if the package is already installed.
    # The output is redirected to /dev/null to keep the console clean.
    if ! python3 -m pip show "$pkg" &>/dev/null; then
        echo "  Installing $pkg..."
        # Use 'python3 -m pip install' to ensure we are using the pip from our
        # activated virtual environment.
        python3 -m pip install "$pkg" || { echo "  ERROR: Failed to install $pkg. Exiting."; exit 1; }
    else
        echo "  $pkg is already installed."
    fi
done


# --- Create .env File for API Keys ---
# This section ensures that a .env file exists for storing secret keys.
# Use the absolute path to avoid ambiguity.
ABS_ENV_FILE="$ABS_PROJECT_DIR/.env"

echo "Checking for .env file..."
# Check if the .env file does NOT exist.
if [ ! -f "$ABS_ENV_FILE" ]; then
    echo "Creating .env file with placeholder API keys at '$ABS_ENV_FILE'."
    # Create the file and add placeholder keys. The user must replace these.
    echo "OPENROUTER_API_KEY=\"your-openrouter-key-here\"" > "$ABS_ENV_FILE"
    echo "LANGSMITH_API_KEY=\"your-langsmith-key-here\"" >> "$ABS_ENV_FILE"
    echo "GOOGLE_API_KEY=\"your-gemini-api-key-here\"" >> "$ABS_ENV_FILE"
else
    echo ".env file already exists. Skipping creation."
fi


# --- Final Instructions ---
# Print a confirmation message and next steps for the user.
echo ""
echo "Setup complete!"
echo "1. Remember to edit '$ABS_ENV_FILE' with your actual API keys (especially GOOGLE_API_KEY)."
echo "2. Activate your venv in new terminal sessions using: source $ACTIVATE_SCRIPT_PATH"

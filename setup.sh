#!/bin/bash

PROJECT_DIR="langgraph-agent"
VENV_SUBDIR_NAME="venv" # Standard venv subdirectory name

ENV_FILE="$PROJECT_DIR/.env" # Still needed for other keys

# Get the directory where the script is located (absolute path)
# This is crucial for correctly building absolute paths later.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Absolute path to the project directory
ABS_PROJECT_DIR="$SCRIPT_DIR/$PROJECT_DIR"

# Variable to hold the absolute path to the activate script
ACTIVATE_SCRIPT_PATH=""

# --- Determine the ACTIVATE_SCRIPT_PATH ---
# Case 1: Project directory already exists
if [ -d "$ABS_PROJECT_DIR" ]; then
    echo "Directory '$ABS_PROJECT_DIR' already exists."

    # Check for flat venv structure first (e.g., bin/activate directly in ABS_PROJECT_DIR)
    if [ -f "$ABS_PROJECT_DIR/bin/activate" ]; then
        echo "Detected existing flat virtual environment structure within '$ABS_PROJECT_DIR'."
        ACTIVATE_SCRIPT_PATH="$ABS_PROJECT_DIR/bin/activate"
    # Check for standard venv subdirectory structure (e.g., venv/bin/activate inside ABS_PROJECT_DIR)
    elif [ -f "$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME/bin/activate" ]; then
        echo "Detected existing standard virtual environment structure within '$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME'."
        ACTIVATE_SCRIPT_PATH="$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME/bin/activate"
    else
        # If project dir exists but no valid activate script, it's an error.
        echo "ERROR: '$ABS_PROJECT_DIR' exists, but no 'bin/activate' script was found (neither flat nor in '$VENV_SUBDIR_NAME/')."
        echo "Please ensure a valid virtual environment exists within '$ABS_PROJECT_DIR'."
        exit 1
    fi
else
    # Case 2: Project directory does NOT exist, create everything new
    echo "Creating directory '$ABS_PROJECT_DIR'..."
    mkdir "$ABS_PROJECT_DIR" || { echo "ERROR: Failed to create directory $ABS_PROJECT_DIR. Exiting."; exit 1; }

    echo "Creating new virtual environment in '$VENV_SUBDIR_NAME' inside '$ABS_PROJECT_DIR'..."
    # Using 'python3 -m venv' explicitly
    (cd "$ABS_PROJECT_DIR" && python3 -m venv "$VENV_SUBDIR_NAME") || \
        { echo "ERROR: Failed to create virtual environment in '$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME'. Exiting."; exit 1; }

    ACTIVATE_SCRIPT_PATH="$ABS_PROJECT_DIR/$VENV_SUBDIR_NAME/bin/activate"
fi

# IMPORTANT: Change directory to the project root *after* determining the activate script's absolute path,
# but *before* sourcing. This ensures subsequent commands (like 'pip') run relative
# to the project's root and that the activate script is correctly sourced with its absolute path.
cd "$ABS_PROJECT_DIR" || { echo "ERROR: Failed to create directory $ABS_PROJECT_DIR. Exiting."; exit 1; }

# --- Deactivate any previously active virtual environment ---
# This helps prevent conflicts if another venv was somehow active from the parent shell.
if type deactivate &>/dev/null; then
    echo "Deactivating any existing virtual environment..."
    deactivate
fi


# --- Activate the virtual environment using its absolute path ---
echo "Activating virtual environment from '$ACTIVATE_SCRIPT_PATH'..."
if ! source "$ACTIVATE_SCRIPT_PATH"; then
    echo "ERROR: Failed to activate virtual environment. Exiting."
    exit 1
fi

# --- DEBUGGING OUTPUT (can be removed later once confirmed) ---
echo "--- DEBUGGING AFTER ACTIVATION ---"
echo "Current working directory: `pwd`"
which python
which python3 # Added to explicitly check python3 as well
which pip
echo "PATH: $PATH"
echo "--- End Debugging ---"


# --- Check and install core Python packages ---
echo "Checking and installing Python packages..."
# Removed guardrails-ai and its specific hub validators
# Updated with termcolor
PYTHON_PACKAGES="langgraph langchain_openai langchain_core langchain-google-genai wikipedia pydantic python-dotenv langsmith jmespath termcolor"

for pkg in $PYTHON_PACKAGES; do
    # Using 'python3 -m pip' explicitly
    if ! python3 -m pip show "$pkg" &>/dev/null; then
        echo "  Installing $pkg..."
        python3 -m pip install "$pkg" || { echo "  ERROR: Failed to install $pkg. Exiting."; exit 1; }
    else
        echo "  $pkg is already installed."
    fi
done

# Removed all Guardrails.ai specific configuration and validator installation logic
# as it's no longer part of this demo.


# --- Check for .env file and create if not found ---
# Use absolute path for .env file for consistency
ABS_ENV_FILE="$ABS_PROJECT_DIR/.env"

echo "Checking for .env file..."
if [ ! -f "$ABS_ENV_FILE" ]; then
    echo "Creating .env file with placeholder API keys at '$ABS_ENV_FILE'."
    echo "OPENROUTER_API_KEY=\"your-openrouter-key-here\"" > "$ABS_ENV_FILE"
    echo "LANGSMITH_API_KEY=\"your-langsmith-key-here\"" >> "$ABS_ENV_FILE"
    echo "GOOGLE_API_KEY=\"your-gemini-api-key-here\"" >> "$ABS_ENV_FILE" # Added Gemini API key placeholder
else
    echo ".env file already exists. Skipping creation."
    # Optionally, you could add logic here to check if GOOGLE_API_KEY is present and warn if not.
fi

echo ""
echo "Setup complete!"
echo "1. Remember to edit '$ABS_ENV_FILE' with your actual API keys (especially GOOGLE_API_KEY)."
echo "2. Activate your venv in new terminal sessions using: source $ACTIVATE_SCRIPT_PATH"

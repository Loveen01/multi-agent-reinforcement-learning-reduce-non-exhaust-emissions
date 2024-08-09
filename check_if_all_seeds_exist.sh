#!/bin/bash
directory="$1" # No spaces around the '='

# Check if the directory argument is provided
if [[ -z "$directory" ]]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

# Array of required file endings
required_endings=(10 15 22 31 55 39 83 49 51 74)
all_exist=true

# Loop through each required file ending
for ending in "${required_endings[@]}"; do
    # Check if a file with the specific ending exists
    if ! ls "$directory"/*"$ending" 1> /dev/null 2>&1; then
        echo "Required file ending with $ending does NOT exist in $directory."
        all_exist=false
    else
        echo "Required file ending with $ending exists in $directory."
    fi
done

if [ "$all_exist" = true ]; then
    echo "All required files exist."
else
    echo "Some required files are missing."
    exit 1
fi
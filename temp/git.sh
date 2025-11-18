#!/bin/bash

# Navigate to the project directory
cd /home/nkj/ws/plantf

# Add all changes to staging
git add .

# Commit changes with a message from command line argument
git commit -m "$1"


# Push changes to the remote repository
git push -u origin main
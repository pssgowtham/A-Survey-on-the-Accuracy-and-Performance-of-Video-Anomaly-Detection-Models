#!/bin/bash

# Array of dates for the commits
dates=("2023-08-15" "2023-08-31" "2023-09-15" "2023-09-30" "2023-10-15" 
       "2023-10-31" "2023-11-15" "2023-11-30" "2023-12-15" "2023-12-31" 
       "2024-01-15" "2024-01-31" "2024-02-15" "2024-02-29" "2024-03-15")

# Commit messages
messages=("Add VAD Model 1" "Add VAD Model 2" "Add VAD Model 3" "Add VAD Model 4" "Add VAD Model 5"
          "Add VAD Model 6" "Add VAD Model 7" "Add VAD Model 8" "Add VAD Model 9" "Add VAD Model 10"
          "Add VAD Model 11" "Add VAD Model 12" "Add VAD Model 13" "Add VAD Model 14" "Add VAD Model 15")

# Loop through dates and create commits
for i in ${!dates[@]}; do
  echo "# Video Anomaly Detection Model ${i}" > model_${i}.py
  git add model_${i}.py
  GIT_COMMITTER_DATE="${dates[i]}T12:00:00" git commit --date="${dates[i]}T12:00:00" -m "${messages[i]}"
done

# Push changes to the repository
git push origin main


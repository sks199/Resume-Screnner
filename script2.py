import os
import csv
import re
from pathlib import Path
import pickle

def categorize_resumes(resume_dir, model):
  """Categorizes resumes in the given directory and outputs results to both directory structures and a CSV file."""

  # Create a dictionary to store the resumes in categories.
  categories = {}

  # Iterate over all the resumes in the directory.
  for resume in os.listdir(resume_dir):
    # Get the resume filename.
    resume_filename = os.path.join(resume_dir, resume)

    # Read the resume file.
    with open(resume_filename,  "r", errors="ignore") as f:
      resume_text = f.read()

    # Classify the resume using the model.
    resume_domain = model.predict(resume_text)

    # Add the resume to the appropriate category in the dictionary.
    if resume_domain not in categories:
      categories[resume_domain] = []
    categories[resume_domain].append(resume_filename)

  # Create a directory for each category.
  for category in categories:
    os.mkdir(category)

  # Iterate over all the categories and move the resumes to the appropriate directories.
  for category in categories:
    for resume in categories[category]:
      os.rename(resume, os.path.join(category, resume))

  # Create a CSV file to store the results.
  with open("results.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Resume Filename", "Domain"])
    for category in categories:
      for resume in categories[category]:
        writer.writerow([resume, category])

if __name__ == "__main__":
  # Get the directory containing the resumes.
  resume_dir = os.path.dirname("D:\\pythonProject\\ML task\\resume_dir")
  

  # Load the model.
  model = pickle.load(open("D:\pythonProject\ML task\saved model\model2.pkl","rb"))

  # Categorize the resumes and output the results.
  categorize_resumes(resume_dir, model)
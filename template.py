import os

def create_project_structure(base_dir="project"):
    # Define the folder and file structure as a dictionary
    structure = {
        "data": ["cifar100", "embeddings"],
        "notebooks": ["exploration.ipynb", "retrieval_demo.ipynb", "generation_demo.ipynb"],
        "src": {
            "models": ["vit.py", "retrieval.py", "generative.py"],
            "utils": ["data_loader.py", "helpers.py"],
            "app.py": None,
            "main.py": None
        },
        "": ["requirements.txt", "README.md", "setup.py"]
    }

    def create_dir(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

    def create_file(path, content="# Placeholder content\n"):
        with open(path, "w") as f:
            f.write(content)
        print(f"Created file: {path}")

    # Create the base directory
    create_dir(base_dir)

    # Create directories and files based on the structure dictionary
    for key, value in structure.items():
        if key == "":
            # Files at the root of base_dir
            for filename in value:
                file_path = os.path.join(base_dir, filename)
                create_file(file_path, f"# {filename}\n")
        elif isinstance(value, list):
            # key is a directory containing a list of subdirectories or files
            dir_path = os.path.join(base_dir, key)
            create_dir(dir_path)
            for item in value:
                # If item has an extension, it's a file; otherwise, a subdirectory
                if "." in item:
                    file_path = os.path.join(dir_path, item)
                    # For notebooks, add minimal valid JSON for a Jupyter Notebook
                    if item.endswith(".ipynb"):
                        notebook_content = (
                            '{\n'
                            ' "cells": [],\n'
                            ' "metadata": {},\n'
                            ' "nbformat": 4,\n'
                            ' "nbformat_minor": 2\n'
                            '}\n'
                        )
                        create_file(file_path, notebook_content)
                    else:
                        create_file(file_path, f"# {item}\n")
                else:
                    # Create subdirectory
                    create_dir(os.path.join(dir_path, item))
        elif isinstance(value, dict):
            # key is a directory and value is another dictionary
            sub_dir = os.path.join(base_dir, key)
            create_dir(sub_dir)
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    # sub_key is a directory containing a list of files
                    sub_sub_dir = os.path.join(sub_dir, sub_key)
                    create_dir(sub_sub_dir)
                    for file in sub_value:
                        file_path = os.path.join(sub_sub_dir, file)
                        create_file(file_path, f"# {file}\n")
                else:
                    # sub_key is a file directly under sub_dir
                    file_path = os.path.join(sub_dir, sub_key)
                    create_file(file_path, f"# {sub_key}\n")

if __name__ == "__main__":
    create_project_structure()

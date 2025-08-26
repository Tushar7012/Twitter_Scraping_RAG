import os
from pathlib import Path

package_name = "public_issue_pipeline"

list_of_files = [
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/components/data_ingestion.py",
    f"src/{package_name}/components/data_processing.py",
    f"src/{package_name}/components/embedding_generation.py",
    f"src/{package_name}/components/db_storage.py",
    f"src/{package_name}/utils/__init__.py",
    f"src/{package_name}/utils/common.py",
    f"src/{package_name}/config/__init__.py",
    f"src/{package_name}/config/configuration.py",
    f"src/{package_name}/pipelines/__init__.py",
    f"src/{package_name}/pipelines/prediction_pipeline.py",
    f"src/{package_name}/entity/__init__.py",
    f"src/{package_name}/entity/config_entity.py",
    f"src/{package_name}/constants/__init__.py",
    ".github/workflows/.gitkeep",
    "main.py",
    ".env",
    ".gitignore",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # create an empty file
        print(f"Created empty file: {filepath}")
    else:
        print(f"{filename} already exists.")
import json
import os

def update_notebook():
    notebook_path = r"c:\Users\ACER\Documents\Stock-Opening-Price-Prediction\notebooks\01_EDA.ipynb"
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    changed = False
    for cell in data.get('cells', []):
        source = cell.get('source', [])
        new_source = []
        for line in source:
            new_line = line
            if "2015-01-01" in line:
                new_line = line.replace("2015-01-01", "2010-01-01")
                changed = True
            if "2015-2019" in line:
                new_line = line.replace("2015-2019", "2012-2019")
                changed = True
            new_source.append(new_line)
        cell['source'] = new_source

    if changed:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print("Successfully updated notebooks/01_EDA.ipynb with the 2010-01-01 start date.")
    else:
        print("No changes were needed in the notebook.")

if __name__ == '__main__':
    update_notebook()

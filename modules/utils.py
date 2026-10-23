# File helpers.

import json
import os

def export_json_data(data, export_path):
    """Write data to export_path as JSON, creating the folder if needed.
    Returns (ok, message)."""
    try:
        parent_dir = os.path.dirname(export_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(export_path, 'w') as f:
            json.dump(data, f, indent=3)
        return True, "Export successful!"
    except Exception as e:
        return False, f"Failed to export: {str(e)}"

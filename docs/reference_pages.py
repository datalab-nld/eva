"""Generates the API Reference pages"""

from pathlib import Path

import mkdocs_gen_files
import yaml

with open("mkdocs.yml","r") as fh:
    config = yaml.safe_load(fh)
    
package_path = config.get("package_path","src")
include_init = config.get("include_init",True)

nav = mkdocs_gen_files.Nav()
exclude = set()

for path in sorted(Path(package_path).rglob("*.py")):
    print(path)
    if str(path) in exclude:
        continue
    module_path = path.relative_to(package_path).with_suffix("")
    doc_path = path.relative_to(package_path).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    
    parts = tuple(module_path.parts)
    
    if parts[-1] == "__init__" and include_init:
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue
    
    nav[parts] = doc_path.as_posix()
    
    with mkdocs_gen_files.open(full_doc_path, "w") as fh:
        ident = ".".join(parts)
        fh.write(f"::: {ident}")
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.txt", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

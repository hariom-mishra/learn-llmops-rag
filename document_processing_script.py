from pathlib import Path
import re

pattern = re.compile(r"^\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}\.\d{3}$")
def clean_transscript_text(transscript: str) -> str:
    lines = transscript.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if pattern.match(line):
            continue

        cleaned.append(line)
    
    return "\n".join(cleaned)


def load_and_clean_transcript(original_dir: Path, new_dir: Path):
    original_path = Path(original_dir)
    new_path = Path(new_dir)

    transscripts = original_path.glob("*.txt")
    for transscript in transscripts:
        file_content = transscript.read_text(encoding="utf-8")
        cleaned_content = clean_transscript_text(file_content)

        filename = transscript.name
        output_path = new_path / filename

        output_path.write_text(data=cleaned_content, encoding="utf-8")

if __name__ == "__main__":
    repo_root = Path()

    original_dir = repo_root / "data" / "raw"
    new_dir = repo_root / "data" / "processed"

    load_and_clean_transcript(original_dir=original_dir, new_dir=new_dir)
import os
from typing import Optional
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

def extraction(file_path: str, output_filename: str, overwrite: bool = True) -> Optional[str]:
    """
    📄 Converts a document to Markdown and saves it.

    Args:
        file_path (str): 📁 Path to the input document.
        output_filename (str): 📝 Path where the Markdown will be saved.
        overwrite (bool): 🔁 Whether to overwrite existing files.

    Returns:
        Optional[str]: ✅ Markdown content on success, or None on failure.
    """

    # ✅ Check if input file exists
    if not os.path.isfile(file_path):
        print(f"❌ [Error] File not found: {file_path}")
        return None

    # ⚠️ If output file exists and overwrite is False, load and return it
    if os.path.exists(output_filename) and not overwrite:
        print(f"ℹ️ [Info] File already exists: {output_filename}. Loading existing Markdown.")
        try:
            with open(output_filename, "r", encoding="utf-8") as f:
                markdown_output = f.read()
            print(f"✅ [Success] Loaded existing Markdown from: {output_filename}")
            return markdown_output, None
        except Exception as e:
            print(f"💥 [Error] Failed to load existing file: {e}")
            return None

    try:
        # 🔄 Convert document to Markdown
        print("🔄 Converting document...")
        result = converter.convert(file_path)
        document = result.document
        markdown_output = document.export_to_markdown()

        # 💾 Save Markdown to file
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown_output)

        print(f"✅ [Success] Markdown saved to: {output_filename}")
        print("🏁 Extraction process completed successfully!")
        return markdown_output, document  # Return Markdown content for consistency

    except Exception as e:
        print(f"💥 [Error] Conversion failed: {e}")
        return None

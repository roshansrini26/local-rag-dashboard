from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("data/1706.03762v7.pdf")
markdown = result.document.export_to_markdown()

with open("docling_output.md", "w") as f:
    f.write(markdown)

print("Done. Output saved to docling_output.md")
print(f"Total length: {len(markdown)} characters")
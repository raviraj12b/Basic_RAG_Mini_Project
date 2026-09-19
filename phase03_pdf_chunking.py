import pymupdf


pdf_path = "data/ML Cheatsheet .pdf"

document = pymupdf.open(pdf_path)

print("Total pages:", len(document))


all_text = ""

for page_number, page in enumerate(document, start=1): # type: ignore
    text =page.get_text()
    print(f"Page {page_number} characters:", len(text))

    all_text += text + "\n"

print("\nTotal extracted characters:")
print(len(all_text))


print("\nFirst 500 characters:")
print(all_text[:500])

chunk_size = 500
overlap = 100

chunks = []
start = 0

while start < len(all_text):
    end = start + chunk_size
    chunk = all_text[start:end]
    chunks.append(chunk)
    start = end - overlap  # Move the start position for the next chunk with overlap

print("\nTotal chunks:")
print(len(chunks))

print("\nFirst chunk:")
print(chunks[0])

print("\nSecond chunk:")
print(chunks[1])

print("\nChunk lengths:")

for i, chunk in enumerate(chunks[:5], start=1):
    print(
        f"Chunk {i}: {len(chunk)} characters"
    )
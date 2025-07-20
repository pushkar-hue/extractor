import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count
import time
import os

def get_title(doc):
    """
    Extracts the document title from metadata or by a robust heuristic.
    """
    if doc.metadata and doc.metadata.get('title'):
        title = doc.metadata['title']
        if title and title.strip():
            return re.sub(r'\s+', ' ', title).strip()

    title_text = "Untitled Document"
    try:
        page = doc[0]
        blocks = page.get_text("dict")["blocks"]
        candidates = {}
        for b in blocks:
            for l in b.get("lines", []):
                if l.get("spans"):
                    line_text = " ".join(s['text'].strip() for s in l['spans'])
                    line_size = round(l['spans'][0]['size'])
                    if line_size not in candidates:
                        candidates[line_size] = []
                    candidates[line_size].append(line_text)
        
        if candidates:
            largest_size = max(candidates.keys())
            title_text = " ".join(candidates[largest_size])
            title_text = re.sub(r'\s+', ' ', title_text).strip()

    except (IndexError, ValueError):
        pass
        
    return title_text if len(title_text) > 4 else "Untitled Document"

def extract_outline_from_toc(doc):
    """
    Extracts the outline from the PDF's built-in Table of Contents (most reliable).
    """
    toc = doc.get_toc()
    if not toc:
        return None
    
    outline = []
    for level, text, page in toc:
        if 1 <= level <= 3:
            outline.append({"level": f"H{level}", "text": text.strip(), "page": page})
            
    return outline if outline else None

def clean_text(text):
    """Clean and validate extracted text."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Check for fragmented text patterns (repeated characters/words)
    if len(text) > 30:  # Only check longer texts
        words = text.split()
        if len(words) > 5:  # Only check if enough words
            # Check if more than 50% of words are repeated
            word_counts = Counter(words)
            repeated_words = sum(1 for count in word_counts.values() if count > 1)
            if repeated_words / len(words) > 0.5:
                return ""
    
    # Filter out obvious garbage text
    if re.search(r'(.)\1{6,}', text):  # 7+ repeated characters
        return ""
    
    # Filter out text with too many special characters
    if text:
        special_char_ratio = len(re.findall(r'[^\w\s\-.,;:()"]', text)) / len(text)
        if special_char_ratio > 0.6:
            return ""
    
    return text

def is_likely_heading(text, font_size, is_bold, body_size, heading_sizes):
    """Determine if text is likely a heading based on multiple criteria."""
    if not text or len(text.strip()) < 2:
        return False
    
    # Clean the text first
    clean = clean_text(text)
    if not clean:
        return False
    
    # More generous length constraints for headings
    word_count = len(clean.split())
    if word_count > 25:  # Increased from 15
        return False
    
    if len(clean) > 200:  # Increased from 150
        return False
    
    # Check for obvious non-heading patterns first (strict filters)
    strict_non_heading_patterns = [
        r'^\$[\d,]+(\.\d{2})?\s*$',  # Just money amounts
        r'^Page\s+\d+\s*$',  # Page numbers
        r'^[^a-zA-Z]*$',  # No letters at all
        r'^[^\w\s]*$',  # Only special characters
    ]
    
    for pattern in strict_non_heading_patterns:
        if re.match(pattern, clean, re.IGNORECASE):
            return False
    
    # Strong positive indicators (these are definitely headings)
    strong_heading_patterns = [
        r'^(Chapter|Section|Part|Appendix|Summary|Background|Introduction|Conclusion|References?|Bibliography)\s*[A-Z0-9]*',
        r'^\d+(\.\d+)*\s+[A-Za-z]',  # "1.1 Introduction"
        r'^[A-Z]{2,}(\s+[A-Z]{2,})*\s*$',  # ALL CAPS headings
        r'^(Abstract|Executive Summary|Table of Contents|Acknowledgments?)',
    ]
    
    for pattern in strong_heading_patterns:
        if re.match(pattern, clean, re.IGNORECASE):
            return True
    
    # Font-based criteria
    significant_font_advantage = font_size > body_size * 1.2  # 20% larger
    moderate_font_advantage = font_size > body_size * 1.05   # 5% larger
    
    # If it has significant font advantage, it's likely a heading
    if significant_font_advantage:
        return True
    
    # If it's bold and moderately larger, likely a heading
    if is_bold and moderate_font_advantage:
        return True
    
    # If it's bold and at least as large as body text, check content
    if is_bold and font_size >= body_size:
        # Title case or sentence case with capitalized first word
        if clean and (clean[0].isupper() or clean.istitle()):
            # Avoid obvious non-headings even when bold
            loose_non_heading_patterns = [
                r'^\d+\.\d+\s*$',  # Just numbers like "3.1"
                r'^[A-Z]\.\d+\s*$',  # Just section refs like "A.1"
                r'^\d{4}[\s-]\d{4}\s*$',  # Date ranges
                r'^\(\s*[^)]*\s*\)\s*$',  # Text in parentheses only
                r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\s*$',  # Dates
            ]
            
            for pattern in loose_non_heading_patterns:
                if re.match(pattern, clean, re.IGNORECASE):
                    return False
            
            return True
    
    # Title case text with moderate font advantage
    if moderate_font_advantage and clean and clean[0].isupper():
        # Check if it looks like a proper title
        words = clean.split()
        if len(words) <= 8 and not re.search(r'[.]{2,}', clean):  # No multiple dots
            return True
    
    return False

def extract_outline_heuristically(doc):
    """
    Extracts outline using improved heuristics with better text cleaning and validation.
    """
    font_counts = Counter()
    all_text_blocks = []
    
    # Pass 1: Collect all text blocks and font information
    for page_num, page in enumerate(doc):
        try:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if not b.get("lines"):
                    continue
                
                # Extract text more carefully, span by span
                block_text_parts = []
                font_sizes = []
                fonts = []
                
                for line in b["lines"]:
                    if not line.get("spans"):
                        continue
                    
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if text:
                            block_text_parts.append(text)
                            font_sizes.append(round(span["size"]))
                            fonts.append(span.get("font", ""))
                
                if block_text_parts:
                    full_text = " ".join(block_text_parts)
                    avg_font_size = round(sum(font_sizes) / len(font_sizes))
                    is_bold = any("bold" in font.lower() for font in fonts)
                    
                    font_counts[avg_font_size] += 1
                    all_text_blocks.append({
                        "text": full_text,
                        "font_size": avg_font_size,
                        "is_bold": is_bold,
                        "page": page_num + 1
                    })
                    
        except Exception:
            continue

    if not font_counts:
        return []

    # Determine body text size (most common)
    try:
        body_text_size = font_counts.most_common(1)[0][0]
    except IndexError:
        return []

    # Get heading sizes (larger than body text)
    heading_sizes = sorted([size for size in font_counts if size > body_text_size], reverse=True)
    
    # Create size-to-level mapping with better thresholds
    size_to_level = {}
    if heading_sizes:
        # More conservative level assignment
        if len(heading_sizes) >= 3:
            # Use top 3 distinct sizes
            size_to_level[heading_sizes[0]] = "H1"
            size_to_level[heading_sizes[1]] = "H2"
            for size in heading_sizes[2:]:
                size_to_level[size] = "H3"
        elif len(heading_sizes) == 2:
            size_to_level[heading_sizes[0]] = "H1"
            size_to_level[heading_sizes[1]] = "H2"
        else:
            size_to_level[heading_sizes[0]] = "H1"

    outline = []
    processed_headings = set()

    # Pass 2: Filter and classify headings
    for block in all_text_blocks:
        text = block["text"]
        font_size = block["font_size"]
        is_bold = block["is_bold"]
        page = block["page"]
        
        # Clean and validate the text
        clean = clean_text(text)
        if not clean:
            continue
        
        # Check if this looks like a heading
        if not is_likely_heading(clean, font_size, is_bold, body_text_size, heading_sizes):
            continue
        
        # Determine heading level
        level = size_to_level.get(font_size)
        if not level:
            # If bold and reasonably sized, make it H3
            if is_bold and font_size >= body_text_size:
                level = "H3"
            else:
                continue
        
        # Avoid duplicates
        heading_key = (page, clean.lower())
        if heading_key not in processed_headings:
            outline.append({
                "level": level,
                "text": clean,
                "page": page
            })
            processed_headings.add(heading_key)

    return outline

def process_single_pdf(pdf_path):
    """Worker function to process one PDF file."""
    try:
        doc = fitz.open(pdf_path)
        
        if doc.page_count > 50:
            doc.close()
            return {"path": pdf_path, "status": "error", "data": {"title": os.path.basename(pdf_path), "outline": [{"level": "H1", "text": "Error: PDF exceeds 50 page limit", "page": 1}]}}

        title = get_title(doc)
        outline = extract_outline_from_toc(doc)
        if not outline:
            outline = extract_outline_heuristically(doc)
        
        doc.close()
        
        return {"path": pdf_path, "status": "success", "data": {"title": title, "outline": outline}}
    except Exception as e:
        return {"path": pdf_path, "status": "error", "data": {"title": os.path.basename(pdf_path), "outline": [{"level": "H1", "text": f"Error processing file: {e}", "page": 1}]}}

def main():
    """Main function to find and process all PDFs in parallel."""
    start_time = time.time()
    input_dir = Path("./input")
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in /app/input.")
        return

    num_processes = min(cpu_count(), len(pdf_files))
    print(f"Found {len(pdf_files)} PDF(s). Starting processing with {num_processes} core(s).")

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_pdf, pdf_files)

    for result in results:
        pdf_path = result["path"]
        output_filename = output_dir / f"{Path(pdf_path).stem}.json"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(result["data"], f, indent=4, ensure_ascii=False)
        
        print(f"[{result['status'].upper()}] Processed '{Path(pdf_path).name}' -> '{output_filename.name}'")

    end_time = time.time()
    print(f"\nProcessing complete in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
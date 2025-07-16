import pdfplumber
import fitz  # PyMuPDF
import os
import re
import google.generativeai as genai
import requests
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import pkg_resources
import base64
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFPaperAnalyzer:
    def __init__(self, pdf_path: str, api_key: str, expertise: str = "research scientist"):
        self.expertise = expertise
        self.pdf_path = Path(pdf_path)
        self.api_key = api_key
        self.sections = {}

        # list of recognized headings and their patterns
        self.section_regexes = {
            'abstract':       r'^\s*(\d+(\.\d+)*)?\s*abstract\s*$',
            'introduction':   r'^\s*(\d+(\.\d+)*)?\s*introduction\s*$',
            'related work':   r'^\s*(\d+(\.\d+)*)?\s*(related work|related-work)\s*$',
            'approach':       r'^\s*(\d+(\.\d+)*)?\s*approach\s*$',
            'methodology':    r'^\s*(\d+(\.\d+)*)?\s*(methodology|methods?)\s*$',
            'results':        r'^\s*(\d+(\.\d+)*)?\s*results?\s*$',
            'discussion':     r'^\s*(\d+(\.\d+)*)?\s*discussion\s*$',
            'conclusion':     r'^\s*(\d+(\.\d+)*)?\s*conclusions?\s*$',
            'references':     r'^\s*(\d+(\.\d+)*)?\s*(references|bibliography)\s*$',
            'appendix':       r'^\s*(\d+(\.\d+)*)?\s*appendix(\s+[a-z0-9]+)?\s*$'
        }

        self.tables = []  # list of (table_data, caption, table_number)
        self.figures = []
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.tables_output_file = os.path.join(self.script_dir, 'extracted_tables.txt')
        self.figures_output_dir = os.path.join(self.script_dir, 'extracted_figures')
        self.figures_metadata_file = os.path.join(self.script_dir, 'extracted_figures.txt')
        self.setup_gemini()

    def setup_gemini(self):
        try:
            genai_version = pkg_resources.get_distribution("google-generativeai").version
            logger.info(f"Using google-generativeai version: {genai_version}")
            genai.configure(api_key=self.api_key)
            logger.info("Gemini API configured successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise

    def extract_text_sections(self) -> None:
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                self.sections = {}
                current_section_title = "Unknown"
                self.sections[current_section_title] = ""

                for page in pdf.pages:
                    lines = (page.extract_text() or "").split('\n')

                    for line in lines:
                        raw_line = line.strip()
                        if not raw_line:
                            self.sections[current_section_title] += "\n"
                            continue

                        heading_num_match = re.match(
                            r'^\s*\d+(\.\d+)*(\s+[A-Z0-9].*)?$',
                            raw_line
                        )

                        is_allcaps = False
                        if (len(raw_line) <= 70 
                            and raw_line == re.sub(r'[^a-zA-Z]', '', raw_line).upper() 
                            and len(raw_line) > 1
                        ):
                            is_allcaps = True

                        if heading_num_match or is_allcaps:
                            current_section_title = raw_line
                            if current_section_title not in self.sections:
                                self.sections[current_section_title] = ""
                        else:
                            self.sections[current_section_title] += line + "\n"

            logger.info(f"Sections found: {list(self.sections.keys())}")
            logger.info("Text sections extracted successfully.")

        except Exception as e:
            logger.error(f"Error extracting text sections: {e}")
            raise

    def extract_tables(self) -> None:
        """Extract tables and their captions from the PDF."""
        try:
            table_dict = {}  # Map table_number -> (table_data, caption)

            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    raw_tables = page.find_tables(
                        table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "intersection_x_tolerance": 5,
                            "intersection_y_tolerance": 5,
                            "snap_tolerance": 2,
                            "join_tolerance": 2
                        }
                    )

                    page_text = page.extract_text() or ''
                    lines = page_text.split('\n')

                    captions_found = [
                        line.strip()
                        for line in lines
                        if re.match(r'^\s*table\s*\d+[.:].*', line, re.IGNORECASE)
                    ]

                    if not raw_tables:
                        continue

                    for tbl in raw_tables:
                        table_data = tbl.extract()
                        if not table_data:
                            continue

                        if len(table_data) < 1 or len(table_data[0]) < 1:
                            continue

                        non_empty_cells = sum(
                            1
                            for row in table_data
                            for cell in row
                            if cell is not None and str(cell).strip()
                        )
                        if non_empty_cells < 4:
                            continue

                        if captions_found:
                            caption = captions_found[0]
                        else:
                            caption = f"Table (Page {page_num+1})"

                        match = re.match(r'^\s*table\s*(\d+)[.:]', caption, re.IGNORECASE)
                        if match:
                            table_number = match.group(1)
                        else:
                            table_number = str(len(table_dict) + 1)

                        if table_number in table_dict:
                            existing_table, existing_caption = table_dict[table_number]
                            existing_rows = {tuple(r) for r in existing_table}
                            for row in table_data:
                                row_tuple = tuple(str(c) if c else "" for c in row)
                                if row_tuple not in existing_rows:
                                    existing_table.append(row)
                                    existing_rows.add(row_tuple)
                        else:
                            table_dict[table_number] = (table_data, caption)

            self.tables = [
                (data, caption, num)
                for num, (data, caption) in sorted(table_dict.items(), key=lambda x: int(x[0]))
            ]

            logger.info(f"Extracted {len(self.tables)} tables.")
            self.save_tables()
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            raise

    def save_tables(self):
        try:
            with open(self.tables_output_file, 'w', encoding='utf-8') as f:
                if not self.tables:
                    f.write("No tables extracted.\n")
                    return
                for i, (table_data, caption, table_number) in enumerate(self.tables, 1):
                    f.write(f"Table {table_number}: {caption}\n")
                    try:
                        table_str = "\n".join([
                            "\t".join(str(cell) if cell else "" for cell in row)
                            for row in table_data
                        ])
                        f.write(f"{table_str}\n\n")
                    except Exception as e:
                        logger.error(f"Error writing table {table_number}: {e}")
                        f.write("[Table data could not be processed]\n\n")
            logger.info(f"Saved tables to {self.tables_output_file}")
        except Exception as e:
            logger.error(f"Error saving tables to file: {e}")
            raise

    def extract_figures(self) -> None:
        """Extract figures/images and their captions from the PDF."""
        try:
            os.makedirs(self.figures_output_dir, exist_ok=True)
            figure_dict = {}
            doc = fitz.open(self.pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                blocks = page.get_text("blocks")

                possible_captions = []
                for block in blocks:
                    block_text = block[4].strip()
                    match = re.match(r'^\s*figure\s*(\d+)[.:](.*)', block_text, re.IGNORECASE)
                    if match:
                        figure_number, caption_text = match.groups()
                        possible_captions.append((int(figure_number), caption_text.strip(), block[:4]))

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    width, height = base_image["width"], base_image["height"]
                    if width < 100 or height < 100:
                        continue
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    img_rect = page.get_image_bbox(img)
                    closest_caption = None
                    min_dist = float('inf')
                    for fig_num, cap_text, bbox in possible_captions:
                        caption_rect = fitz.Rect(bbox)
                        distance = min(
                            abs(img_rect.y0 - caption_rect.y1),
                            abs(img_rect.y1 - caption_rect.y0)
                        )
                        if distance < min_dist:
                            min_dist = distance
                            closest_caption = (fig_num, cap_text)

                    if not closest_caption:
                        continue

                    fig_num, cap_text = closest_caption
                    figure_number = str(fig_num)
                    if figure_number not in figure_dict:
                        image_filename = f"figure_page{page_num + 1}_img{img_index}.{image_ext}"
                        image_path = os.path.join(self.figures_output_dir, image_filename)
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        figure_dict[figure_number] = (image_path, cap_text)

            doc.close()

            self.figures = [
                (path, caption, num)
                for num, (path, caption) in sorted(figure_dict.items(), key=lambda x: int(x[0]))
            ]
            logger.info(f"Extracted {len(self.figures)} figures.")
            self.save_figures_metadata()
        except Exception as e:
            logger.error(f"Error extracting figures: {e}")
            raise

    def save_figures_metadata(self):
        try:
            with open(self.figures_metadata_file, 'w', encoding='utf-8') as f:
                if not self.figures:
                    f.write("No figures extracted.\n")
                    return
                for i, (image_path, caption, figure_number) in enumerate(self.figures, 1):
                    image_filename = os.path.basename(image_path)
                    f.write(f"Figure {figure_number}: {image_filename}, Caption: {caption}\n")
            logger.info(f"Saved figure metadata to {self.figures_metadata_file}")
        except Exception as e:
            logger.error(f"Error saving figure metadata to file: {e}")
            raise

    def prepare_context(self) -> str:
        context = "Analysis of Academic Paper\n\n"

        typical_order = [
            "abstract", "introduction", "related work", "approach",
            "methodology", "results", "discussion", "conclusion",
            "references", "appendix"
        ]
        sorted_keys = []
        for h in typical_order:
            if h in self.sections and self.sections[h].strip():
                sorted_keys.append(h)

        for h in self.sections:
            if h not in sorted_keys:
                sorted_keys.append(h)

        for heading in sorted_keys:
            content = self.sections[heading].strip()
            if content:
                context += f"## {heading.capitalize()}\n{content}\n\n"

        if self.tables:
            context += "## Tables\n"
            for table_data, caption, table_number in self.tables:
                context += f"### Table {table_number}: {caption}\n"
                try:
                    table_str = "\n".join([
                        "\t".join(str(cell) if cell is not None else "" for cell in row)
                        for row in table_data
                    ])
                    context += f"{table_str}\n\n"
                except Exception as e:
                    logger.error(f"Error processing table {table_number}: {e}")
                    context += "[Table data could not be processed]\n\n"

        if self.figures:
            context += "## Figures\n"
            for img_path, caption, figure_number in self.figures:
                context += f"### Figure {figure_number}: {caption}\n"
                context += f"[Image description: {os.path.basename(img_path)}]\n\n"

        return context

    def analyze_paper_quality(self) -> str:
        try:
            context = self.prepare_context()
            prompt = (
                f"You are an expert academic reviewer with the following expertise: {self.expertise}. Below is the extracted content from an academic paper, "
                "including sections, tables, and figure captions. Analyze the quality based on:\n"
                "1. Novelty: The novelty and innovativeness of contributed solutions, problem formulations, methodologies, theories, and/or evaluations, i.e., the extent to which the paper is sufficiently original with respect to the state-of-the-art.\n"
                "2. Rigor: The soundness, clarity, and depth of a technical or theoretical contribution, and the level of thoroughness and completeness of an evaluation.\n"
                "3. Relevance: The significance and/or potential impact of the research on the field of software engineering.\n"
                "4. Verifiability and Transparency: The extent to which the paper includes sufficient information to understand how an innovation works; to understand how data was obtained, analyzed, and interpreted; and how the paper supports independent verification or replication of the paper's claimed contributions. Any artifacts attached to or linked from the paper will be checked by one reviewer.\n"
                "5. Presentation: The clarity of the exposition in the paper.\n"
                "Provide a structured assessment with headings for each criterion and an overall quality rating (1-10).\n\n"
                f"Paper Content:\n{context}"
            )
            response = genai.generate_content(
                model="models/gemini-2.0-flash",
                contents=[{"parts": [{"text": prompt}]}],
                generation_config={"temperature": 0.9}
            )
            response_text = ""
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    response_text += part.text
            logger.info("Paper quality analysis completed.")
            return response_text
        except AttributeError as e:
            logger.warning(f"SDK method failed: {e}. Falling back to HTTP request.")
            return self.analyze_paper_quality_http(prompt)
        except Exception as e:
            logger.error(f"Error analyzing paper quality: {e}")
            raise

    def analyze_paper_quality_http(self, prompt: str) -> str:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            data = {
                "generationConfig": {                   
                    "temperature": 0.9
                },
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
            text = ""
            for candidate in result.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    text += part.get("text", "")
            logger.info("Paper quality analysis completed via HTTP.")
            return text
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            raise

    def run(self) -> str:
        try:
            logger.info(f"Starting analysis for {self.pdf_path}")
            self.extract_text_sections()
            self.extract_tables()
            self.extract_figures()
            result = self.analyze_paper_quality()
            logger.info("Analysis completed successfully.")
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
        finally:
            # Clean up extracted images
            for img_path, _, _ in self.figures:
                if os.path.exists(img_path):
                    os.remove(img_path)


def main(pdf_path: str) -> None:
    try:
        if not os.path.exists(pdf_path) or not pdf_path.lower().endswith('.pdf'):
            raise ValueError("Invalid PDF file path or file is not a PDF.")
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        analyzer = PDFPaperAnalyzer(pdf_path, api_key)
        result = analyzer.run()
        print("Paper Quality Analysis Result:\n")
        print(result)
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 pdf_analyzer.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    main(pdf_path)
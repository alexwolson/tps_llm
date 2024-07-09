import argparse
import os
from openai import OpenAI
from pathlib import Path
import pymupdf
from rich.console import Console
from rich.progress import Progress

console = Console()

def pdf_page_iterator(pdf_path: Path, num_pages: int):
    """
    A generator function that yields files with chunks of pages from a PDF.

    Args:
    pdf_path (str): Path to the PDF file.
    num_pages (int): Number of pages to yield at a time.

    Yields:
    Path: Path to a PDF file containing the specified number of pages.
    """
    if num_pages == -1:
        # Extract all pages
        yield pdf_path
        return

    document = pymupdf.open(pdf_path)
    total_pages = document.page_count
    console.print(f'Total pages in the document: {total_pages}')

    # Iterate through the document in chunks of num_pages
    for start_page in range(0, total_pages, num_pages):
        end_page = min(start_page + num_pages, total_pages)

        # Create a temporary file
        temp_pdf_path = f"temp_pdf_pages_{start_page + 1}_to_{end_page}.pdf"

        # Create a new PDF containing the chunk of pages
        temp_document = pymupdf.open()  # Create a new PDF in memory
        for page_num in range(start_page, end_page):
            temp_document.insert_pdf(document, from_page=page_num, to_page=page_num)

        # Save the chunk to the temporary file
        temp_document.save(temp_pdf_path)
        temp_document.close()

        yield Path(temp_pdf_path)


def extract_from_one_document(document_path: Path, model: str, prompt: str, client: OpenAI, num_pages: int) -> list:
    """
    Extracts references from a single document.

    Args:
    document_path (Path): Path to the document.
    model (str): The model to use for extracting references.
    prompt (str): The prompt to use for extracting references.
    client (OpenAI): The OpenAI client.
    num_pages (int): Number of pages to process at a time.

    Returns:
    list: A list of extracted references.
    """
    references = []

    with Progress() as progress:
        task = progress.add_task(f"[cyan]Processing {document_path.name}...", total=document_path.stat().st_size)

        # Iterate through the pages of the document
        for page_path in pdf_page_iterator(document_path, num_pages):
            progress.update(task, advance=os.path.getsize(page_path))

            # Upload the pages to OpenAI
            message_file = client.files.create(file=page_path, purpose='assistants')

            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "attachments": [
                            {"file_id": message_file.id, "tools": [{"type": "file_search"}]}
                        ],
                    }
                ]
            )

            assistant = client.beta.assistants.create(
                name="Technology Purchases",
                instructions=prompt,
                model=model,
                tools=[{"type": "file_search"}],
            )

            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id, assistant_id=assistant.id
            )

            messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
            try:
                message_content = messages[0].content[0].text.value
            except IndexError:
                console.print(f'No references found in {page_path.name}.')
                console.print(message_content)
                raise Exception('No references found in the document.')

            references.append(message_content)

            # Delete the temporary file
            page_path.unlink()

    return references


def extract_references(input_path: Path, output_path: Path, model: str, prompt: str, client: OpenAI, num_pages: int):
    """
    Extracts references from a set of documents and writes them to an output file.

    Args:
    input_path (Path): Path to the input document or directory.
    output_path (Path): Path to the output file.
    model (str): The model to use for extracting references.
    prompt (str): The prompt to use for extracting references.
    client (OpenAI): The OpenAI client.
    num_pages (int): Number of pages to process at a time.

    Returns:
    None
    """
    files = []

    if input_path.is_dir():
        files = list(input_path.rglob('*.pdf'))
    elif input_path.is_file():
        files = [input_path]

    console.log(f'Found {len(files)} files to process.')

    references = []
    for file in files:
        console.print(f'Processing file: {file}')
        references.extend(extract_from_one_document(file, model, prompt, client, num_pages))

    with open(output_path, 'w') as f:
        for reference in references:
            f.write(reference + '\n')

    console.log(f'All references written to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract references to technology purchases from a document or set of documents.')
    parser.add_argument('-i', '--input', type=Path,
                        help='Path to the document or directory containing documents to extract references from.')
    parser.add_argument('-o', '--output', type=Path, help='Path to the output file to write the extracted references to.',
                        default='references.csv')
    parser.add_argument('-m', '--model', type=str, help='The model to use for extracting references.',
                        default='gpt-3.5-turbo')
    parser.add_argument('-k', '--api_key', type=str, help='The OpenAI API key to use for extracting references.',
                        default='openai.key')
    parser.add_argument('-p', '--prompt', type=Path, help='Path to the prompt to use for extracting references.',
                        default='prompt.txt')
    parser.add_argument('-n', '--pages', type=int,
                        help='The number of pages to extract references from at a time. -1 for all pages.', default=10)
    parser.add_argument('-g', '--organization', type=str, help='The organization to use for extracting references.',
                        default='org-l0QXnTWrsY221IPu8QIF1k1H')
    parser.add_argument('-r', '--project', type=str, help='The project to use for extracting references.',
                        default='proj_bD68pqv8EEnCgfp2ujfsoJpJ')

    args = parser.parse_args()

    # Initialize OpenAI Client
    with open(args.api_key, 'r') as f:
        client = OpenAI(api_key=f.read().strip(), organization=args.organization, project=args.project)

    # Load Prompt
    with open(args.prompt, 'r') as f:
        prompt = f.read().strip()

    extract_references(args.input, args.output, args.model, prompt, client, args.pages)

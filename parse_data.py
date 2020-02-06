from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import xml.etree.ElementTree as ET

root_dir = Path('./data/NKJP-PodkorpusMilionowy-1.2')
out_dir = Path('./data/text')


def main():
    text_files = [
        directory / 'text.xml'
        for directory in root_dir.iterdir()
        if directory.is_dir()
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(get_sentences_from_xml, text_files)


def get_sentences_from_xml(file_path):
    root = ET.parse(file_path).getroot()
    sentences = [elem.text for elem in root.findall('.//{http://www.tei-c.org/ns/1.0}ab')]

    out_dir.mkdir(exist_ok=True)
    outfile = out_dir / f'{file_path.parent.name}.txt'
    with open(outfile, 'w') as output_file:
        output_file.write(' '.join(sentences))


if __name__ == '__main__':
    main()

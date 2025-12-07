"""
data_clear.py

@author: chensong
@date: 2025年12月7日

Simple text cleaning script used for preparing the local dataset `华盖集.txt`.

This module performs light heuristic cleaning steps commonly used when preparing
text corpora for NLP/LLM experiments:

- Read the source file from `./data/华盖集.txt`.
- Remove special chapter-separator blocks that match a custom pattern.
- Collapse multiple whitespace characters into a single space.
- Remove short paragraphs (heuristic: lines whose length is < 10 characters).
- Write the cleaned output to `./data/华盖集_cleaned.txt`.

Notes:
- This file intentionally uses simple, readable regex heuristics. For production
    preprocessing consider using more robust tokenization and sentence-splitting
    libraries (e.g. jieba/PKUSeg/StanfordNLP) depending on the language.

Usage:
        Run the file from the repository root (where `./data/华盖集.txt` exists):

                python LLM/data_clear.py

The script prints a short message when finished and writes the cleaned file.
"""

import re


# Path to the dataset directory (relative to repository root)
base_data_path = "./data/"


def main():
    """Main entrypoint: read, clean, and save the dataset.

    The function reads `华盖集.txt` from `base_data_path`, applies a sequence
    of light cleaning operations, and writes `华盖集_cleaned.txt`.
    """

    # Read the raw text file. Ensure the file exists in `./data/`.
    with open(base_data_path + "华盖集.txt", "r", encoding="utf-8") as f:
        context = f.read()

    # 1) Remove custom chapter-separator blocks and collapse surrounding noise.
    #    The original data contained separators matching a pattern like:
    #    lines including several `※` characters followed by `&` (legacy marker).
    #    The code below walks line-by-line and skips content between the
    #    separator and the next chapter heading (lines starting with '##').
    lines = context.splitlines()  # split into lines for simple stateful filtering
    clean_lines = []
    skip_mode = False

    for line in lines:
        stripped = line.strip()

        # Enter skip mode when encountering the separator marker. This is a
        # heuristic targeted at noisy metadata present in the raw file.
        if re.match(r'^.*※.*※.*※.*&', stripped):
            skip_mode = True
            continue

        # Exit skip mode when a chapter heading (starting with '##') is found.
        if skip_mode and re.match(r'^##.*$', stripped):
            skip_mode = False
            continue

        # Only keep lines when not in skip mode.
        if not skip_mode:
            clean_lines.append(line)

    # Re-join the filtered lines back into a single string
    context = '\n'.join(clean_lines)

    # 2) Collapse multiple whitespace characters (including newlines and tabs)
    #    into a single space to normalize spacing across the file. This makes
    #    downstream tokenization and sentence detection more predictable.
    cleaned_context = re.sub(r'\s+', ' ', context, flags=re.MULTILINE)

    # Optional: remove unusual special characters but keep common Chinese
    # punctuation. The following commented pattern demonstrates how to remove
    # non-word non-space characters while preserving Chinese punctuation.
    # cleaned_context = re.sub(r"[^\w\s，。！？、“”‘’]", '', cleaned_context, flags=re.MULTILINE)

    # 3) Remove very short paragraphs (heuristic): if a substring between
    #    newline boundaries is under 10 characters we drop it. This helps remove
    #    stray markers or tiny fragments that are not useful for training.
    cleaned_context = re.sub(r'(?<=\n)(.{1,10})(?=\n)', '', cleaned_context, flags=re.MULTILINE)

    # Save the cleaned output to a new file so the original remains intact.
    out_path = base_data_path + "华盖集_cleaned.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned_context)

    print("数据清洗完成，已保存到 华盖集_cleaned.txt")


if __name__ == '__main__':
    main()

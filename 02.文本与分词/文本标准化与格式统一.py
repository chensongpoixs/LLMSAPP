# @author: chensong
# @file: 红楼梦.py
# @time: 2025-12-16 00:00
# @desc: 爬取《红楼梦》所有章节标题和内容
# 功能：爬取《红楼梦》所有章节标题和内容
# 目标网站：https://hongloumeng.5000yan.com
'''
文本标准化与格式统一
Docstring for 02.文本与分词.文本标准化与格式统一
'''



import re


# Path to the dataset directory (relative to repository root)
base_data= "./data/hongloumeng.txt";
base_basesetdata = "./data/baseset_hongloumeng.txt";


def main():
    """Main entrypoint: read, clean, and save the dataset.
    The function reads `hongloumeng.txt` from `base_data_path`, applies a sequence
    """

    # Read the raw text file. Ensure the file exists in `./data/`.
    with open(base_data , "r", encoding="utf-8") as f:
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
        if re.match(r'^.*※.*※.*※.* .*&', stripped):
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
    out_path = base_basesetdata;
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned_context)

    print("数据清洗完成，已保存到 {}".format(base_basesetdata));


if __name__ == '__main__':
    main()

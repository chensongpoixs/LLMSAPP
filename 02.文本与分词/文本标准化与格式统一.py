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


# 红楼梦的数据集位置
base_data= "./data/hongloumeng.txt";
# 文本统一化 
base_basesetdata = "./data/baseset_hongloumeng.txt";


def main():
    """
    读取红楼梦的数据 进行文本标准化与格式统一的操作 

    1. 清洗
    2. 去重
    3. tokenizer
    """

    # 读取数据集
    with open(base_data , "r", encoding="utf-8") as f:
        context = f.read()

   
    # 删除行
    lines = context.splitlines()  # split into lines for simple stateful filtering
    clean_lines = []
    skip_mode = False
    # 遍历 行的数据
    for line in lines:
        stripped = line.strip()

        # 过滤不规范数据删除了
        if re.match(r'^.*※.*※.*※.* .*&', stripped):
            skip_mode = True
            continue

        # 删除## 的
        if skip_mode and re.match(r'^##.*$', stripped):
            skip_mode = False
            continue

        # Only keep lines when not in skip mode.
        if not skip_mode:
            clean_lines.append(line)

    # Re-join the filtered lines back into a single string
    context = '\n'.join(clean_lines)

    # 当前 => 
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

"""
scraper.py

@author: chensong
@date: 2025年12月7日

一个简易的网络爬虫示例，用于从示例站点 `mingzhuxiaoshuo.com` 下载连载小说（或其他分页内容），
并将每个章节保存到 `./data/` 下的单一文本文件中。

此脚本作为教育示例，展示了使用 requests 发起请求、使用 BeautifulSoup 解析 HTML、
简单的错误处理以及遵守礼貌性爬取（请求间休眠）。

注意事项：
- 在爬取前请检查目标站点的 robots.txt 及服务条款，确保允许抓取。
- 请尊重频率限制，不要对服务器进行高频或并发请求。
"""

import requests
from bs4 import BeautifulSoup
import time
import os


# 目标页面（连载小说的目录页）。若要抓取同站点的其它书籍，可修改此 URL。
url = "https://www.mingzhuxiaoshuo.com/jinxiandai/111/"
base_url = "https://www.mingzhuxiaoshuo.com/"


# 保存爬取数据的目录，脚本会确保该目录存在。
base_data_path = "./data/"


def ensure_data_dir(path: str):
    """若目录不存在则创建数据保存目录。"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_soup(url: str, headers: dict, timeout: int = 10):
    """请求指定 URL 并返回解析后的 BeautifulSoup 对象。

    说明：目标站点使用 GBK 编码，函数会显式设置编码后再解析。若遇到网络错误
    或者返回码不是 200，则返回 None。
    """
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        print(f"网络错误: 无法请求 {url}: {e}")
        return None

    # 许多中文网站使用 GBK 编码，这里在解析前显式设置编码。
    resp.encoding = 'gbk'

    if resp.status_code != 200:
        print(f"错误：无法获取页面 {url}, 状态码：{resp.status_code}")
        return None

    return BeautifulSoup(resp.text, 'html.parser')


def main():
    """主要爬取流程：发现章节、遍历并保存章节内容。"""

    ensure_data_dir(base_data_path)

    # 使用简单的请求头模拟浏览器访问。
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; scraper/1.0; +https://example.com)'
    }

    print("爬虫开始工作......")

    # 请求目录页并构建 BeautifulSoup 对象。
    soup = get_soup(url, headers)
    if soup is None:
        return

    # 页面结构示例：书名通常位于 <h1> 元素中。
    title_tag = soup.find('h1')
    if not title_tag:
        print("错误：未能在主页中找到 <h1> 标题元素，无法确定书名。")
        return

    title = title_tag.text.strip()
    output_filename = f"{title}.txt"
    print(f"爬取的数据集的名子是：{output_filename}")

    # 在目录页中查找章节链接列表，站点使用 class='list' 的容器承载链接。
    chapter_list = soup.find('div', class_='list')
    if not chapter_list:
        print("错误：在主页上未找到 class='list' 的章节列表")
        return

    chapter_tags = chapter_list.find_all('a')

    # 构建章节信息（标题、链接）的字典列表，便于后续迭代抓取。
    chapter_infos = []
    for tag in chapter_tags:
        # tag['href'] sometimes is a relative path; join with base_url.
        chapter_url = base_url + tag['href'].lstrip('/')
        chapter_title = tag.text.strip()
        chapter_infos.append({'title': chapter_title, 'url': chapter_url})

    print(f"共找到{len(chapter_infos)}个章节！")

    # 以追加方式打开单个输出文件，按顺序写入每章内容。将所有章节合并到一个文件
    # 更便于后续处理和训练语料的构建。
    out_path = os.path.join(base_data_path, output_filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, chapter in enumerate(chapter_infos):
            t = chapter['title']
            u = chapter['url']
            print(f"正在爬取{i+1}/{len(chapter_infos)}章: {t}, url: {u}")

            chapter_soup = get_soup(u, headers)
            if chapter_soup is None:
                print(f"警告：无法访问章节页面 {u}")
                continue

            # 章节正文通常位于 id='content' 的 div 中。
            content_div = chapter_soup.find('div', id='content')
            if content_div:
                # 使用 get_text 并以换行符为分隔，尽量保留段落格式。
                content_text = content_div.get_text(separator='\n').strip()
                f.write(f"##{t}\n\n")
                f.write(content_text)
                f.write("\n\n\n")
            else:
                print(f"警告：在页面 {u} 未找到正文内容！")

            # 礼貌性等待：每次请求后短暂停顿，避免对目标服务器造成压力。
            time.sleep(1)

    print("所有章节下载完成！")


if __name__ == '__main__':
    main()

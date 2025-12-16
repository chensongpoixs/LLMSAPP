# -*- coding: utf-8 -*-
# @Author  : chensong
# @File    : 红楼梦.py
# @Time    : 2025-12-16 01:00:00
# @Desc    : 爬取《红楼梦》所有章节标题和内容
# 功能：爬取《红楼梦》所有章节标题和内容
# 目标网站：https://hongloumeng.5000yan.com/
import requests
from bs4 import BeautifulSoup


# 红楼梦目录页地址
base_url = "https://hongloumeng.5000yan.com/";


# 数据保存路径
save_path = "./data/hongloumeng.txt"

def book_spider(url):
    """
    爬取红楼梦文本信息
    :param url: 小说目录页网址
    :return:
    """
    # 1. 进行UA伪装，模拟浏览器访问
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    print("开始爬取《红楼梦》全文...");
    # 2. 发送请求
    page_text = requests.get(url=url, headers=headers)
    page_text.encoding = page_text.apparent_encoding # 自动获取编码防止乱码
    page_text = page_text.text
    # print("目录页爬取成功！page_tex:", page_text);
    # 3. 解析目录页，获取所有章节的链接和标题
    soup = BeautifulSoup(page_text, 'lxml')



    print("==========soup:", soup);



    # 选择器定位到所有包含章节链接的<a>标签
    aTagList = soup.select('div > ul > li.p-2 > a');
    print("==========aTagList:", aTagList);
    # return;
    titleList = [i.text for i in aTagList] # 章节标题列表
    #urlList = ["https://www.shicimingju.com" + i["href"] for i in aTagList] # 补全为完整链接
    urlList = [ i["href"] for i in aTagList] # 补全为完整链接
    # 4. 创建文件并写入总标题
    with open(save_path, 'w', encoding='utf-8') as fp:
        fp.write("红楼梦\n")
    # 5. 遍历每一章，调用函数下载内容
    for chp in zip(titleList, urlList):
        write_chapter(chp)
    print("《红楼梦》全文爬取完成！")

def write_chapter(content_list):
    """
    提取单个章节内容并追加写入文件
    :param content_list: 包含（标题, 链接）的元组
    :return:
    """
    title, url = content_list
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    # 请求章节详情页
    page_text = requests.get(url=url, headers=headers, timeout=10)
    page_text.encoding = page_text.apparent_encoding
    page_text = page_text.text
    # 解析章节正文内容
    soup = BeautifulSoup(page_text, 'lxml')
    content = soup.select('div > div > div.grap')  # 定位到正文内容的<p>标签列表
    txt = ""
    for i in content:
        txt += i.text
    # 将章节标题和内容追加到文件
    with open(save_path, 'a', encoding='utf-8') as fp:
        fp.write("{}".format('\n\n' + title + '\n'));
        fp.write(txt + '\n');
    print(f"已下载: {title}");

if __name__ == '__main__': 
    book_spider(base_url)


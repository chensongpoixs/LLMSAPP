import requests, json;
import time;


url = "http://localhost:8899/v1/chat/completions"
headers = {"Content-Type": "application/json"}
# 设置目标题目数量
TOTAL_QUESTIONS = 100
# 每次请求获取的题目数 (建议 20-50 之间，避免超出单次输出 Token 限制)
BATCH_SIZE = 5 

# 输出文件名
OUTPUT_FILE = "math_problems_1000.json"

 

# ---------------- 主程序 ----------------
def get_data_from_llama_cpp_data():
    # client = genai.Client(api_key=API_KEY)
    
    all_data = [] # 用来存所有的题目
    
    print(f"开始获取 {TOTAL_QUESTIONS} 道纯数学题 (无答案)...")

    while len(all_data) < TOTAL_QUESTIONS:
        # 计算还要多少道
        needed = TOTAL_QUESTIONS - len(all_data)
        current_batch_size = min(BATCH_SIZE, needed)
        
        print(f"正在请求... (当前进度: {len(all_data)}/{TOTAL_QUESTIONS})")

        # 提示词：明确告诉它不要答案
        prompt = (
            f"请给我 {current_batch_size} 道小学 4-6 年级的数学题。"
            "要求：\n"
            "1. 只要题目，绝对不要答案，也不要选项。\n"
            "2. 题目类型包含计算、应用题、几何。\n"
            "3. 题目描述要清晰。"
        )
        print(f"prompt:{prompt}");
        try:
           
            data = {
                #"model": "gpt-3.5-turbo",
                "model": "ggml-org/gemma-3-1b-it-GGUF",
                "messages": [
                    {"role": "system", "content": "小学4~6年级的数学题目的具体内容，不包含答案"},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "tools": [
                {
                  "type": "function",
                  "function": {
                    "name": "get_grade_info",  
                    "description": "小学4~6年级 数学题目的具体内容，不包含答案",
                    "parameters": {  
                      "type": "object",
                      "properties": {
                        "grade": {
                          "type": "string",
                          "description": "年级，如 '四年级','五年级','六年级'"
                        },
                        "content": {
                          "type": "string",
                          "description": "数学题目的具体内容,不包含答案"
                        } 
                      },
                      "required": ["grade", "content"]
                    }
                  }
                }
              ],
              "tool_choice": {"type": "function", "function": {"name": "get_grade_info"}}  
            }
            response = requests.post(url, headers=headers, json=data)
            print(json.dumps(response.json(), indent=2, ensure_ascii=False));
            # 1. 解析整个llama.cpp的响应
            full_response = json.loads(json.dumps(response.json(), indent=2, ensure_ascii=False))

            # 2. 提取工具调用参数
            tool_call = full_response["choices"][0]["message"]["tool_calls"][0]
            function_name = tool_call["function"]["name"]
            arguments_dict = json.loads(tool_call["function"]["arguments"])

            # 3. 根据函数名执行不同逻辑
            if function_name == "get_grade_info":
                grade = arguments_dict["grade"]
                content = arguments_dict["content"]
                # ... 执行你的业务逻辑
                print(f"收到请求: 为{grade}年级生成题目 - {content}")
                all_data.append({"grade": grade, "content": content});
            # "choices"
            # res_data = json.loads(json.dumps(response.json(), indent=2, ensure_ascii=False));
            # content = res_data["choices"][0]["message"]["content"]
            # print(f"response.choices[0].message.content:{content}");
            # if  content:
            #     all_data.append(content);
            #if response.parsed:
                # 把这一批题目加到总列表里
            #    for item in response.parsed:
            #        all_data.append(item.model_dump())
            
        except Exception as e:
            print(f"出错重试: {e}")
            time.sleep(2)
            continue
            
       # time.sleep(1) # 稍微歇一下，防止请求太快

    # ---------------- 保存文件 ----------------
    # 构造成你想要的格式 { "questions": [ ... ] }
    final_json = {
        "questions": all_data
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print(f"完成！已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    get_data_from_llama_cpp_data()
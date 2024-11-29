import gradio as gr
import requests
import base64
import json
import io
from PIL import Image
import numpy as np
from PIL import ImageDraw
import logging
from config import (
    API_URL, 
    API_KEY, 
    DEFAULT_MODEL,
    MAX_IMAGE_SIZE_MB,
    MAX_RETRIES,
    RETRY_DELAY
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加常量配置
DESCRIPTION = "图片元素定位工具 - 基于Claude Vision实现的GUI元素定位"
_SYSTEM_PROMPT = """基于页面截图，我会给出文本描述，你需要给出对应的位置。
坐标表示元素的可点击位置[x, y]，是截图上的相对坐标，范围从0到1。"""

def compress_image(image, max_size_mb=MAX_IMAGE_SIZE_MB):
    """压缩图片到指定大小以下"""
    logging.info("开始压缩图片")
    # 初始质量
    quality = 95
    output = io.BytesIO()
    
    # 压缩循环
    while quality > 5:
        output.seek(0)
        output.truncate()
        image.save(output, format='JPEG', quality=quality)
        if len(output.getvalue()) <= max_size_mb * 1024 * 1024:
            break
        quality -= 5
    
    logging.info("图片压缩完成")
    return output.getvalue()

def draw_box(image, x1, y1, x2, y2, color='red', width=2):
    """在图像上绘制方框标记"""
    logging.info(f"在图像上绘制方框，坐标: ({x1}, {y1}, {x2}, {y2})")
    draw = ImageDraw.Draw(image)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return image

def analyze_image_gradio(image, target_element, model_name, test_mode=False):
    """Gradio界面调用的主函数"""
    logging.info(f"开始分析图片，目标元素: {target_element}，使用模型: {model_name}")
    
    # 删除原有的硬编码配置
    max_retries = MAX_RETRIES
    retry_delay = RETRY_DELAY
    
    compressed_image = compress_image(image)
    image_data = base64.b64encode(compressed_image).decode('utf-8')
    
    # 获取图片尺寸
    width, height = image.size
    
    # 修改第一步的提示词，专注于获取注意力区域
    prompt = f"""Please find the region in the desktop GUI screenshot that matches the requirement of "{target_element}".
Image dimensions: width={width} pixels, height={height} pixels
Notes:
- This is a screenshot of a desktop GUI interface
- A larger attention region bounding box is required, ensuring the region is complete
- The coordinate system uses relative coordinates, with x and y values ranging from 0 to 1
- Only return the format: [x1, y1, x2, y2], representing the coordinates of the top-left and bottom-right corners of the region
- For example: [0.1, 0.2, 0.4, 0.5]"""

    url = API_URL
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
    
    # 在成功获取注意力区域后，添加第二步精确定位
    def locate_target_in_region(region_image, target_element):
        """在裁剪区域内精确定位目标"""
        logging.info("开始在区域内精确定位目标")
        
        # 修改提示词，使其更明确且避免特殊字符
        region_prompt = f"""In this cropped region, please precisely locate the position that meets the requirement of "{target_element}".
        Do not return coordinates in blank areas. Please strictly follow the format below:
Return coordinates in the following format:
[x, y]
where x and y are values between 0 and 1, representing relative positions."""

        region_payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": region_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(compress_image(region_image)).decode('utf-8')}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }

        max_parse_retries = 3  # 最大重试次数
        
        for parse_attempt in range(max_parse_retries):
            try:
                response = requests.post(url, headers=headers, json=region_payload)
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 改进坐标解析逻辑
                try:
                    import re
                    pattern = r"\[([\d.]+)[,\s]+([\d.]+)\]"
                    matches = re.search(pattern, content)
                    if matches:
                        coordinates = [float(matches.group(1)), float(matches.group(2))]
                        # 验证坐标范围
                        if all(0 <= coord <= 1 for coord in coordinates):
                            return coordinates
                    logging.warning(f"第 {parse_attempt + 1} 次尝试解析坐标失败: {content}")
                    if parse_attempt < max_parse_retries - 1:
                        continue
                except Exception as e:
                    logging.error(f"第 {parse_attempt + 1} 次坐标解析出错: {str(e)}")
                    if parse_attempt < max_parse_retries - 1:
                        continue
            except Exception as e:
                logging.error(f"第 {parse_attempt + 1} 次精确定位失败: {str(e)}")
                if parse_attempt < max_parse_retries - 1:
                    continue
                return None
        
        logging.error("达到最大重试次数，坐标解析失败")
        return None

    # 在成功获取第一次坐标后添加验证逻辑
    if test_mode:
        marked_image = image.copy()
        # 使用固定的相对坐标
        rel_x, rel_y = 0.57, 0.75
        x = int(rel_x * width)
        y = int(rel_y * height)
        marked_image = draw_box(marked_image, x, y, x, y)
        logging.info(f"测试模式，返回固定坐标: x={rel_x:.2f}, y={rel_y:.2f}")
        return marked_image, f"找到 {target_element} 的相对位置：x={rel_x:.2f}, y={rel_y:.2f}"
    else:
        # 原有的坐标获取逻辑
        for attempt in range(max_retries):
            try:
                logging.info(f"尝试获取坐标，第 {attempt + 1} 次")
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                # 添加响应内容验证
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    logging.error(f"API响应解析失败: {response.text}")
                    return image, f"API响应格式错误: {str(e)}"
                
                # 验证响应结构
                if not isinstance(result, dict) or 'choices' not in result:
                    logging.error(f"API响应格式不正确: {result}")
                    return image, f"API响应格式不正确: {result}"
                    
                if not result['choices'] or not isinstance(result['choices'], list):
                    logging.error(f"API响应中没有choices数据: {result}")
                    return image, f"API响应中没有choices数据: {result}"
                    
                content = result["choices"][0].get("message", {}).get("content")
                if not content:
                    logging.error(f"API响应中没有content数据: {result}")
                    return image, f"API响应中没有content数据: {result}"
                
                # 修改坐标处理部分
                try:
                    coordinates = None
                    try:
                        coordinates = eval(content.strip())
                    except:
                        import re
                        pattern = r"\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]"
                        matches = re.search(pattern, content)
                        if matches:
                            coordinates = [float(matches.group(i)) for i in range(1, 5)]
                    
                    if coordinates and isinstance(coordinates, list) and len(coordinates) == 4:
                        # 裁剪注意力区域
                        region_x1 = int(coordinates[0] * width)
                        region_y1 = int(coordinates[1] * height)
                        region_x2 = int(coordinates[2] * width)
                        region_y2 = int(coordinates[3] * height)
                        
                        region_image = image.crop((region_x1, region_y1, region_x2, region_y2))
                        
                        # 在区域内精确定位
                        target_coords = locate_target_in_region(region_image, target_element)
                        
                        if target_coords:
                            # 将区域内的相对坐标转换为原图坐标
                            abs_x = region_x1 + target_coords[0] * (region_x2 - region_x1)
                            abs_y = region_y1 + target_coords[1] * (region_y2 - region_y1)
                            
                            # 绘制结果
                            marked_image = image.copy()
                            # 绘制注意力区域
                            marked_image = draw_box(marked_image, region_x1, region_y1, region_x2, region_y2, color='blue', width=2)
                            # 绘制目标点
                            point_size = 5
                            marked_image = draw_box(marked_image, 
                                                  int(abs_x - point_size), int(abs_y - point_size),
                                                  int(abs_x + point_size), int(abs_y + point_size),
                                                  color='red', width=2)
                            
                            rel_x, rel_y = abs_x / width, abs_y / height
                            return marked_image, f"找到 {target_element} 的位置：x={rel_x:.2f}, y={rel_y:.2f}"
                    
                    logging.error(f"坐标格式不正确: {content}")
                    return image, f"坐标格式不正确: {content}"
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logging.error(f"处理过程中出现错误: {error_details}")
                    return image, f"处理过程中出现错误: {str(e)}"
                
            except requests.exceptions.HTTPError as e:
                # 添加更详细的错误信息
                error_msg = f"HTTP错误: {str(e)}"
                try:
                    error_msg += f"\n响应内容: {e.response.text}"
                except:
                    pass
                logging.error(error_msg)
                if e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                return image, error_msg
            except requests.exceptions.RequestException as e:
                logging.error(f"网络请求错误: {str(e)}")
                return image, f"网络请求错误: {str(e)}"
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                logging.error(f"处理过程中出现错误: {error_details}")
                return image, f"处理过程中出现错误: {str(e)}"
        
        logging.error("达到最大重试次数，请稍后再试")
        return image, "达到最大重试次数，请稍后再试"

# 创建Gradio界面
def create_interface():
    """创建更专业的Gradio界面"""
    with gr.Blocks(title="图片元素定位工具", theme=gr.themes.Default()) as demo:
        # 添加更详细的说明
        gr.Markdown(f"""
        ## {DESCRIPTION}
        
        ### 使用说明
        1. 上传一张GUI界面截图
        2. 输入要查找的元素描述
        3. 点击"查找位置"按钮
        4. 系统将在图片上标记出元素位置
        """)
        
        # 使用状态管理
        state_image_path = gr.State(value=None)
        
        with gr.Row():
            with gr.Column(scale=3):
                image_input = gr.Image(
                    type="pil",
                    label="上传GUI截图",
                    elem_id="image_input"
                )
                text_input = gr.Textbox(
                    label="要查找的元素描述",
                    placeholder="例如：点击关闭按钮",
                    elem_id="text_input"
                )
                model_input = gr.Textbox(
                    label="模型名称",
                    value=DEFAULT_MODEL,  # 使用配置中的默认模型
                    elem_id="model_input"
                )
                
                # 添加示例
                gr.Examples(
                    examples=[
                        ["./examples/chrome.png", "点击搜索框"],
                        ["./examples/chrome.png", "点击设为默认"],
                        ["./examples/ios_setting.png", "Click Do Not Disturb"],
                    ],
                    inputs=[image_input, text_input],
                    examples_per_page=3
                )
            
            with gr.Column(scale=4):
                image_output = gr.Image(type="pil", label="标记结果")
                output_text = gr.Textbox(label="查找结果")
                
                with gr.Row():
                    submit_btn = gr.Button("查找位置", variant="primary")
                    clear_btn = gr.Button("清除", variant="secondary")
                    retry_btn = gr.Button("重试", variant="secondary")

        submit_btn.click(
            fn=lambda img, txt, model: analyze_image_gradio(img, txt, model, test_mode=False),
            inputs=[image_input, text_input, model_input],
            outputs=[image_output, output_text]
        )
    
    return demo

def main():
    """主函数入口"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    
    demo = create_interface()
    demo.queue(api_open=False).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()

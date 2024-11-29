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

def calculate_confidence(coordinates, target_element, region_size, model_output):
    """计算检测结果的置信度
    
    Args:
        coordinates: 归一化的坐标 [x1, y1, x2, y2]
        target_element: 目标元素描述
        region_size: 图像尺寸 (width, height)
        model_output: 模型输出的文本
    
    Returns:
        float: 置信度分数 (0-1)
    """
    confidence = 0.0
    width, height = region_size
    
    # 1. 基础坐标检查 (0.25分)
    if all(0 <= c <= 1 for c in coordinates):
        confidence += 0.25
    
    # 2. 区域大小合理性检查 (0.25分)
    region_width = coordinates[2] - coordinates[0]
    region_height = coordinates[3] - coordinates[1]
    pixel_area = region_width * width * region_height * height
    
    # 根据目标类型调整期望的区域大小
    if any(keyword in target_element.lower() for keyword in ['按钮', 'button', '图标', 'icon']):
        # 按钮和图标通常较小
        if 1000 <= pixel_area <= 10000:
            confidence += 0.25
    elif any(keyword in target_element.lower() for keyword in ['输入框', 'input', '文本框', 'textbox']):
        # 输入框通常较大
        if 5000 <= pixel_area <= 50000:
            confidence += 0.25
    else:
        # 其他元素使用中等大小判断
        if 2000 <= pixel_area <= 30000:
            confidence += 0.25
    
    # 3. 宽高比检查 (0.25分)
    aspect_ratio = region_width / region_height if region_height > 0 else 0
    if any(keyword in target_element.lower() for keyword in ['按钮', 'button']):
        # 按钮通常是矩形
        if 2.0 >= aspect_ratio >= 1.5:
            confidence += 0.25
    elif any(keyword in target_element.lower() for keyword in ['图标', 'icon']):
        # 图标通常接近正方形
        if 1.2 >= aspect_ratio >= 0.8:
            confidence += 0.25
    else:
        # 其他元素使用通用判断
        if 3.0 >= aspect_ratio >= 0.3:
            confidence += 0.25
    
    # 4. 位置合理性检查 (0.25分)
    center_x = (coordinates[0] + coordinates[2]) / 2
    center_y = (coordinates[1] + coordinates[3]) / 2
    
    # 检查是否在合理的屏幕区域内
    if 0.1 <= center_x <= 0.9 and 0.1 <= center_y <= 0.9:
        confidence += 0.25
    
    return min(1.0, max(0.0, confidence))

def classify_element(image, target_element, model_name):
    """对目标元素进行分类"""
    logging.info(f"开始对元素进行分类: {target_element}")
    
    compressed_image = compress_image(image)
    image_data = base64.b64encode(compressed_image).decode('utf-8')
    
    classification_prompt = f"""请分析图片中符合"{target_element}"描述的元素可能属于哪种GUI元素类型。
仅从以下类别中选择一个最匹配的：
1. button (按钮)
2. icon (图标)
3. input (输入框)
4. text (文本)
5. link (链接)
6. checkbox (复选框)
7. radio (单选框)
8. dropdown (下拉框)
9. menu (菜单)
10. other (其他)

只返回对应的英文类别名称，例如: button"""

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": classification_prompt
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
        "max_tokens": 50
    }

    try:
        response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload)
        response.raise_for_status()
        result = response.json()
        element_type = result["choices"][0]["message"]["content"].strip().lower()
        logging.info(f"元素分类结果: {element_type}")
        return element_type
    except Exception as e:
        logging.error(f"元素分类失败: {str(e)}")
        return "other"

def get_element_specific_prompt(element_type, target_element):
    """根据元素类型生成特定的提示词"""
    type_specific_prompts = {
        "button": "Note that buttons are typically rectangular areas that may contain text or icons",
        "icon": "Note that icons are typically square areas that may be a single graphical symbol",
        "input": "Note that input fields are typically long rectangular areas that may have placeholder text",
        "text": "Note the boundaries of the text area and avoid including other elements",
        "link": "Note that links typically have underlines or special colors",
        "checkbox": "Note that checkboxes are typically small squares that may have label text",
        "radio": "Note that radio buttons are typically small circles that may have label text", 
        "dropdown": "Note that dropdown menus typically have a small triangle icon",
        "menu": "Note that menu items are typically arranged horizontally or vertically",
        "other": "Carefully observe the visual characteristics of the target element"
    }
    
    return type_specific_prompts.get(element_type, type_specific_prompts["other"])

def draw_grid(image, grid_size=10):
    """在图像上绘制网格和坐标标记
    
    Args:
        image: PIL Image对象
        grid_size: 网格数量（将图像分成 grid_size x grid_size 的网格）
    
    Returns:
        PIL Image对象
    """
    width, height = image.size
    draw = ImageDraw.Draw(image)
    
    # 计算网格间距
    x_step = width / grid_size
    y_step = height / grid_size
    
    # 绘制网格线
    for i in range(grid_size + 1):
        x = i * x_step
        y = i * y_step
        
        # 绘制垂直线
        draw.line([(x, 0), (x, height)], fill='rgba(255,255,255,128)', width=1)
        # 绘制水平线
        draw.line([(0, y), (width, y)], fill='rgba(255,255,255,128)', width=1)
        
        # 添加坐标标记
        if i < grid_size:
            # X轴坐标
            draw.text((x + x_step/2, height - 20), 
                     f'{(i+0.5)/grid_size:.1f}', 
                     fill='white', 
                     anchor='mm')
            # Y轴坐标
            draw.text((10, y + y_step/2), 
                     f'{(i+0.5)/grid_size:.1f}', 
                     fill='white', 
                     anchor='mm')
    
    return image

def analyze_image_gradio(image, target_element, model_name, test_mode=False, force_continue=False):
    """修改主分析函数"""
    # 从配置文件导入常量
    max_retries = MAX_RETRIES
    retry_delay = RETRY_DELAY
    
    # 首先对元素进行分类
    element_type = classify_element(image, target_element, model_name)
    # 获取特定类型的提示词
    type_specific_prompt = get_element_specific_prompt(element_type, target_element)
    
    # 在原始图像上添加网格
    working_image = image.copy()
    working_image = draw_grid(working_image)
    # 保存临时图像用于后续分析
    temp_image_path = "temp_grid_image.jpg"
    working_image.save(temp_image_path)
    # 获取图片尺寸
    width, height = image.size
    
    # 更新提示词，告诉模型关于网格的信息
    prompt = f"""Please find the region in the desktop GUI screenshot that matches the requirement of "{target_element}".
The image has been overlaid with a 10x10 grid system, with coordinates marked along the axes.
Please use these grid lines to provide more accurate coordinates.
Element type: {element_type}
{type_specific_prompt}
Image dimensions: width={width} pixels, height={height} pixels
Notes:
- This is a screenshot of a desktop GUI interface with coordinate grid overlay
- Use the grid lines and coordinates to provide more precise locations
- The coordinate system uses relative coordinates, with x and y values ranging from 0 to 1
- Only return the format: [x1, y1, x2, y2], representing the coordinates of the top-left and bottom-right corners
- For example: [0.1, 0.2, 0.4, 0.5]"""

    # 使用添加了网格的图像进行分析
    image_data = base64.b64encode(compress_image(working_image)).decode('utf-8')
    
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
        
        # 优化提示词，使其更加精确
        region_prompt = f"""请在图像中精确定位与描述"{target_element}"相关的中心点位置。注意：
1. 必须返回元素的精确中心点，而不是周围区域
2. 坐标格式为 [x, y]，其中x和y是0到1之间的相对坐标
3. 如果看到多个可能的目标，请选择最可能的一个
4. 如果目标不清晰，请返回最佳估计位置
5. 避免返回空白区域或边缘位置

请严格按照以下格式只返回坐标：[x, y]，如果找不到，则返回图像中心点[0.5,0.5]"""

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
        return marked_image, f"找到 {target_element} 的相对位置：x={rel_x:.2f}, y={rel_y:.2f}", gr.Row.update(visible=False)
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
                    return image, f"API响应格式错误: {str(e)}", gr.Row.update(visible=False)
                
                # 验证响应结构
                if not isinstance(result, dict) or 'choices' not in result:
                    logging.error(f"API响应格式不正确: {result}")
                    return image, f"API响应格式不正确: {result}", gr.Row.update(visible=False)
                    
                if not result['choices'] or not isinstance(result['choices'], list):
                    logging.error(f"API响应中没有choices数据: {result}")
                    return image, f"API响应中没有choices数据: {result}", gr.Row.update(visible=False)
                    
                content = result["choices"][0].get("message", {}).get("content")
                if not content:
                    logging.error(f"API响应中没有content数据: {result}")
                    return image, f"API响应中没有content数据: {result}", gr.Row.update(visible=False)
                
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
                        # 计算置信度
                        confidence = calculate_confidence(
                            coordinates=coordinates,
                            target_element=target_element,
                            region_size=(width, height),
                            model_output=content
                        )
                        
                        # 如果置信度太低，返回警告
                        if confidence < 0.4 and not force_continue:
                            return (
                                image, 
                                f"识别结果可信度较低 ({confidence:.2f})，是否继续？", 
                                gr.Row.update(visible=True)
                            )
                        
                        # 裁剪注意力区域
                        region_x1 = int(coordinates[0] * width)
                        region_y1 = int(coordinates[1] * height)
                        region_x2 = int(coordinates[2] * width)
                        region_y2 = int(coordinates[3] * height)
                        
                        region_image = image.crop((region_x1, region_y1, region_x2, region_y2))
                        
                        # 在区域内精确定位
                        target_coords = locate_target_in_region(region_image, target_element)
                        
                        if target_coords:
                            # 添加结果验证
                            abs_x = region_x1 + target_coords[0] * (region_x2 - region_x1)
                            abs_y = region_y1 + target_coords[1] * (region_y2 - region_y1)
                            
                            # 1. 验证点击位置是否在有效区域内
                            min_distance_to_edge = 10  # 像素
                            if (abs_x < min_distance_to_edge or abs_x > width - min_distance_to_edge or
                                abs_y < min_distance_to_edge or abs_y > height - min_distance_to_edge):
                                logging.warning("定位结果太靠近边缘，进行调整")
                                abs_x = max(min_distance_to_edge, min(width - min_distance_to_edge, abs_x))
                                abs_y = max(min_distance_to_edge, min(height - min_distance_to_edge, abs_y))
                            
                            # 2. 缩小标记大小以提高精确度
                            point_size = 3  # 改小标记点的大小
                            
                            # 3. 使用不同颜色区分注意力区域和目标点
                            marked_image = image.copy()
                            # 绘制注意力区域
                            marked_image = draw_box(marked_image, region_x1, region_y1, region_x2, region_y2, color='blue', width=2)
                            # 绘制目标点
                            marked_image = draw_box(marked_image, 
                                                  int(abs_x - point_size), int(abs_y - point_size),
                                                  int(abs_x + point_size), int(abs_y + point_size),
                                                  color='red', width=2)
                            
                            rel_x, rel_y = abs_x / width, abs_y / height
                            return marked_image, f"找到 {target_element} 的位置：x={rel_x:.2f}, y={rel_y:.2f} (置信度: {confidence:.2f})", gr.Row.update(visible=False)
                    
                    logging.error(f"坐标格式不正确: {content}")
                    return image, f"坐标格式不正确: {content}", gr.Row.update(visible=False)
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    logging.error(f"处理过程中出现错误: {error_details}")
                    return image, f"处理过程中出现错误: {str(e)}", gr.Row.update(visible=False)
                
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
                return image, error_msg, gr.Row.update(visible=False)
            except requests.exceptions.RequestException as e:
                logging.error(f"网络请求错误: {str(e)}")
                return image, f"网络请求错误: {str(e)}", gr.Row.update(visible=False)
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                logging.error(f"处理过程中出现错误: {error_details}")
                return image, f"处理过程中出现错误: {str(e)}", gr.Row.update(visible=False)
        
        logging.error("达到最大重试次数，请稍后再试")
        return image, "达到最大重试次数，请稍后再试", gr.Row.update(visible=False)

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

        # 添加问答接口
        with gr.Row():
            question_input = gr.Textbox(
                label="输入问题",
                placeholder="在这里输入你的问题",
                elem_id="question_input"
            )
            answer_output = gr.Textbox(label="回答")

            question_btn = gr.Button("获取回答", variant="primary")

        # 添加确认继续按钮
        with gr.Row(visible=False) as confirm_row:
            confirm_btn = gr.Button("确认继续", variant="primary")
        
        # 修改提交按钮的点击事件
        submit_btn.click(
            fn=lambda img, txt, model: analyze_image_gradio(img, txt, model, force_continue=False),
            inputs=[image_input, text_input, model_input],
            outputs=[image_output, output_text, confirm_row]
        )
        
        # 添加确认按钮的点击事件
        confirm_btn.click(
            fn=lambda img, txt, model: analyze_image_gradio(img, txt, model, force_continue=True),
            inputs=[image_input, text_input, model_input],
            outputs=[image_output, output_text, confirm_row]
        )

        # 连接问答接口
        question_btn.click(
            fn=answer_question,
            inputs=[question_input],
            outputs=[answer_output]
        )

    return demo

def answer_question(question):
    """使用Claude API回答问题"""
    logging.info(f"收到问题: {question}")
    
    # 构建Claude API请求
    prompt = f"""You are a helpful assistant. Answer the following question: "{question}"."""
    
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 150
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # 解析Claude API的响应
        content = result["choices"][0].get("message", {}).get("content")
        if content:
            return content.strip()
        else:
            logging.error("Claude API响应中没有content数据")
            return "抱歉，我无法回答这个问题。"
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP错误: {str(e)}")
        return "抱歉，无法连接到Claude API。"
    except requests.exceptions.RequestException as e:
        logging.error(f"网络请求错误: {str(e)}")
        return "抱歉，网络请求出错。"
    except Exception as e:
        logging.error(f"处理过程中出现错误: {str(e)}")
        return "抱歉，处理请求时出错。"

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
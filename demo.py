from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import gradio as gr
import logging
from PIL import Image, ImageDraw
import io
import re
from transformers import BitsAndBytesConfig

# 配置常量
DESCRIPTION = "图片元素定位工具 - 基于Qwen2VL实现的GUI元素定位"
MAX_IMAGE_SIZE_MB = 5
MAX_RETRIES = 3

# 修改模型加载部分
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# 加载模型和处理器
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto", quantization_config=quantization_config
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# 确保模型处于评估模式
model.eval()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compress_image(image, max_size_mb=MAX_IMAGE_SIZE_MB):
    """压缩图片到指定大小以下"""
    logging.info("开始压缩图片")
    quality = 95
    output = io.BytesIO()
    
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

def get_region_coordinates(image, target_element):
    """获取目标区域坐标，并返回坐标和模型输出"""
    width, height = image.size
    
    # 优化提示词
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": f"""Task: Locate the precise region containing "{target_element}" in this GUI screenshot.

Requirements:
1. Focus on interactive elements like buttons, text fields, links, and UI controls
2. Consider both text labels and associated visual elements
3. Include sufficient context around the target element
4. Avoid empty spaces and irrelevant areas

Image specs: {width}x{height} pixels

Return ONLY the normalized coordinates [x1, y1, x2, y2] where:
- (x1, y1): top-left corner
- (x2, y2): bottom-right corner
- All values between 0 and 1"""
                }
            ]
        }
    ]

    # 添加重试机制
    for attempt in range(MAX_RETRIES):
        try:
            # 使用processor处理消息
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 处理图像输入
            inputs = processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(model.device)
            
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            model_output = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            try:
                pattern = r"\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]"
                matches = re.search(pattern, model_output)
                if matches:
                    coords = [float(matches.group(i)) for i in range(1, 5)]
                    # 验证坐标有效性
                    if all(0 <= c <= 1 for c in coords) and coords[0] < coords[2] and coords[1] < coords[3]:
                        return coords, model_output
            except Exception as e:
                logging.error(f"坐标解析错误: {str(e)}")
                continue
        except Exception as e:
            logging.warning(f"第{attempt + 1}次尝试失败: {str(e)}")
            continue
    
    return None, None

def ensure_minimum_size(image, min_size=64):
    """确保图像尺寸不小于最小值，必要时进行放大"""
    width, height = image.size
    if width < min_size or height < min_size:
        # 计算需要的缩放比例
        scale = max(min_size / width, min_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def locate_target_in_region(region_image, target_element):
    """在裁剪区域内精确定位目标"""
    # 图像预处理优化
    region_image = process_image_for_detection(region_image)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": region_image
                },
                {
                    "type": "text",
                    "text": f"""Task: Find the exact click position for "{target_element}" in this region.

Requirements:
1. Focus on the most clickable part of the element
2. For buttons: target the center
3. For text fields: target the input area
4. For links: target the text
5. Avoid padding and margins

Return ONLY normalized coordinates [x, y] (values between 0-1)"""
                }
            ]
        }
    ]

    # 使用processor处理消息
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 处理图像输入
    inputs = processor(
        text=text,
        images=region_image,  # 直接传入图像
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    content = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    try:
        pattern = r"\[([\d.]+)[,\s]+([\d.]+)\]"
        matches = re.search(pattern, content)
        if matches:
            coordinates = [float(matches.group(1)), float(matches.group(2))]
            if all(0 <= coord <= 1 for coord in coordinates):
                return coordinates
    except Exception as e:
        logging.error(f"坐标解析错误: {str(e)}")
    return None

def resize_to_480p(image):
    """将图像调整为480p分辨率，保持宽高比"""
    target_height = 480
    aspect_ratio = image.size[0] / image.size[1]
    target_width = int(target_height * aspect_ratio)
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

def process_image_for_detection(image):
    """优化图像预处理"""
    # 确保最小尺寸
    image = ensure_minimum_size(image, min_size=64)
    
    # 增强对比度
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    
    # 锐化
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.1)
    
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
    
    # 1. 基础坐标检查 (0.2分)
    if all(0 <= c <= 1 for c in coordinates):
        confidence += 0.2
    
    # 2. 区域大小合理性检查 (0.2分)
    region_width = coordinates[2] - coordinates[0]
    region_height = coordinates[3] - coordinates[1]
    pixel_area = region_width * width * region_height * height
    
    # 根据目标类型调整期望的区域大小
    if any(keyword in target_element.lower() for keyword in ['按钮', 'button', '图标', 'icon']):
        # 按钮和图标通常较小
        if 1000 <= pixel_area <= 10000:
            confidence += 0.2
    elif any(keyword in target_element.lower() for keyword in ['输入框', 'input', '文本框', 'textbox']):
        # 输入框通常较大
        if 5000 <= pixel_area <= 50000:
            confidence += 0.2
    else:
        # 其他元素使用中等大小判断
        if 2000 <= pixel_area <= 30000:
            confidence += 0.2
    
    # 3. 宽高比检查 (0.15分)
    aspect_ratio = region_width / region_height if region_height > 0 else 0
    if any(keyword in target_element.lower() for keyword in ['按钮', 'button']):
        # 按钮通常是矩形
        if 2.0 >= aspect_ratio >= 1.5:
            confidence += 0.15
    elif any(keyword in target_element.lower() for keyword in ['图标', 'icon']):
        # 图标通常接近正方形
        if 1.2 >= aspect_ratio >= 0.8:
            confidence += 0.15
    else:
        # 其他元素使用通用判断
        if 3.0 >= aspect_ratio >= 0.3:
            confidence += 0.15
    
    # 4. 位置合理性检查 (0.15分)
    center_x = (coordinates[0] + coordinates[2]) / 2
    center_y = (coordinates[1] + coordinates[3]) / 2
    
    # 检查是否在合理的屏幕区域内
    if 0.1 <= center_x <= 0.9 and 0.1 <= center_y <= 0.9:
        confidence += 0.15
    
    # 5. 模型输出文本分析 (0.3分)
    # 检查模型输出中的确定性指标
    certainty_keywords = ['确定', '明确', 'found', 'located', 'identified', 'exact']
    uncertainty_keywords = ['可能', '也许', 'probably', 'might', 'unclear', 'ambiguous']
    
    certainty_score = 0
    for keyword in certainty_keywords:
        if keyword in model_output.lower():
            certainty_score += 0.1
    
    for keyword in uncertainty_keywords:
        if keyword in model_output.lower():
            certainty_score -= 0.05
    
    confidence += min(0.3, max(0, certainty_score))
    
    return min(1.0, max(0.0, confidence))

def analyze_image_with_qwen(image, target_element, test_mode=False):
    """使用Qwen2VL模型分析图片"""
    try:
        # 图像预处理
        image = process_image_for_detection(image)
        
        # 调整图像尺寸
        original_size = image.size
        resized_image = resize_to_480p(image)
        scale_x = original_size[0] / resized_image.size[0]
        scale_y = original_size[1] / resized_image.size[1]
        
        width, height = resized_image.size
        
        if test_mode:
            marked_image = resized_image.copy()
            rel_x, rel_y = 0.57, 0.75
            x = int(rel_x * width)
            y = int(rel_y * height)
            marked_image = draw_box(marked_image, x, y, x, y)
            return marked_image, f"找到 {target_element} 的相对位置：x={rel_x:.2f}, y={rel_y:.2f}"
        
        # 获取区域坐标和模型输出
        coordinates, model_output = get_region_coordinates(resized_image, target_element)
        if not coordinates:
            return image, "无法识别目标区域"
        
        # 计算置信度
        confidence = calculate_confidence(
            coordinates=coordinates,
            target_element=target_element,
            region_size=(width, height),
            model_output=model_output
        )
        
        # 如果置信度太低，尝试其他尺度
        if confidence < 0.6:
            scales = [0.75, 1.25]  # 尝试其他尺度
            for scale in scales:
                scaled_size = (int(width * scale), int(height * scale))
                scaled_image = resized_image.resize(scaled_size, Image.Resampling.LANCZOS)
                
                new_coordinates, new_model_output = get_region_coordinates(scaled_image, target_element)
                if new_coordinates:
                    new_confidence = calculate_confidence(
                        coordinates=new_coordinates,
                        target_element=target_element,
                        region_size=scaled_size,
                        model_output=new_model_output
                    )
                    
                    if new_confidence > confidence:
                        confidence = new_confidence
                        coordinates = new_coordinates
                        resized_image = scaled_image
                        width, height = scaled_size
                        model_output = new_model_output
        
        # 如果最终置信度仍然太低，返回警告
        if confidence < 0.4:
            return resized_image, f"识别结果可信度较低 ({confidence:.2f})，请确认"
        
        # 继续处理高置信度的结果
        region_x1 = int(coordinates[0] * width)
        region_y1 = int(coordinates[1] * height)
        region_x2 = int(coordinates[2] * width)
        region_y2 = int(coordinates[3] * height)
        
        # 裁剪注意力区域
        region_image = resized_image.crop((region_x1, region_y1, region_x2, region_y2))
        
        # 在区域内精确定位
        target_coords = locate_target_in_region(region_image, target_element)
        
        if target_coords:
            # 添加坐标平滑处理
            abs_x = smooth_coordinate(region_x1 + target_coords[0] * (region_x2 - region_x1))
            abs_y = smooth_coordinate(region_y1 + target_coords[1] * (region_y2 - region_y1))
            
            # 绘制结果
            marked_image = resized_image.copy()
            # 绘制注意力区域
            marked_image = draw_box(marked_image, region_x1, region_y1, region_x2, region_y2, color='blue', width=2)
            # 绘制目标点
            point_size = 5
            marked_image = draw_box(marked_image, 
                                  int(abs_x - point_size), int(abs_y - point_size),
                                  int(abs_x + point_size), int(abs_y + point_size),
                                  color='red', width=2)
            
            # 计算原始图像中的相对坐标
            rel_x, rel_y = (abs_x * scale_x) / original_size[0], (abs_y * scale_y) / original_size[1]
            return marked_image, f"找到 {target_element} 的位置：x={rel_x:.2f}, y={rel_y:.2f} (置信度: {confidence:.2f})"
        
        return resized_image, "无法精确定位目标元素"
    except Exception as e:
        logging.error(f"处理过程出错: {str(e)}")
        return image, "处理过程出现错误，请重试"

def smooth_coordinate(coord):
    """平滑坐标值，避免抖动"""
    return round(coord * 2) / 2  # 四舍五入到0.5的倍数

def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="图片元素定位工具", theme=gr.themes.Default()) as demo:
        gr.Markdown(f"""
        ## {DESCRIPTION}
        
        ### 使用说明
        1. 上传一张GUI界面截图
        2. 输入要查找的元素描述
        3. 点击"查找位置"按钮
        4. 系统将在图片上标记出元素位置
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                image_input = gr.Image(type="pil", label="上传GUI截图")
                text_input = gr.Textbox(label="要查找的元素描述", placeholder="例如：点击关闭按钮")
                
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
                
                submit_btn = gr.Button("查找位置", variant="primary")
        
        submit_btn.click(
            fn=analyze_image_with_qwen,
            inputs=[image_input, text_input],
            outputs=[image_output, output_text]
        )
    
    return demo

def main():
    """主函数入口"""
    demo = create_interface()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    main()

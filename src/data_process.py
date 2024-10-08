import hashlib
import io
import os
import json
from collections import defaultdict
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from datasets import concatenate_datasets
from tqdm import tqdm


def is_cell_in_object(cell_bbox, object_bbox, alpha=0.1):
    """检查 cell 的 bbox 是否在 object 的 bbox 内或边界上"""
    x1, y1, w1, h1 = cell_bbox
    x2, y2, w2, h2 = object_bbox
    return (x1 >= x2 and x1 + w1 <= x2 + w2 + alpha and
            y1 >= y2 and y1 + h1 <= y2 + h2 + alpha)


def sort_cells(cells):
    """对 cells 进行排序，先按 y 坐标（从上到下），再按 x 坐标（从左到右）"""
    return sorted(cells, key=lambda c: (c['bbox'][1], c['bbox'][0]))


def process_sample(sample):
    """处理单个样本"""
    objects = sample['objects']
    cells = sample['cells']

    matched_objects = []

    for obj in objects:
        obj_bbox = obj['bbox']
        obj_cells = []

        for cell in cells:
            cell_bbox = cell['bbox']
            if is_cell_in_object(cell_bbox, obj_bbox):
                obj_cells.append({
                    'bbox': cell_bbox,
                    'text': cell['text']
                })

        # 对匹配到的 cells 进行排序
        sorted_cells = sort_cells(obj_cells)

        # 将排序后的文本添加到 obj 对象中
        obj['text'] = ' '.join([cell['text'] for cell in sorted_cells])
        obj['cells'] = sorted_cells

        matched_objects.append({
            'object': obj,
            'matched_cells': sorted_cells
        })

    # 将处理结果添加到样本中
    sample['matched_objects'] = matched_objects
    return sample


def is_intersect(box1, box2, axis='both'):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    if axis == 'x':
        return x1 < x2 + w2 and x1 + w1 > x2
    elif axis == 'y':
        return y1 < y2 + h2 and y1 + h1 > y2
    elif axis == 'both':
        return (x1 < x2 + w2 and x1 + w1 > x2) and (y1 < y2 + h2 and y1 + h1 > y2)
    else:
        raise ValueError("Invalid axis value. Must be 'x', 'y', or 'both'.")


def is_similar_column(bbox1, bbox2, x_threshold=5, width_threshold=0.1):
    """
    判断两个bbox是否可能属于同一列
    """
    x1, _, w1, _ = bbox1
    x2, _, w2, _ = bbox2

    x_diff = abs(x1 - x2)
    width_diff = abs(w1 - w2) / max(w1, w2)

    return x_diff <= x_threshold and width_diff <= width_threshold


def determine_column_layout(sample):
    objects = sample['objects']
    bboxes = [each['bbox'] for each in objects if each['category_id'] in (0, 3, 6, 7, 8, 9)]
    bboxes = [list(map(round, bbox)) for bbox in bboxes]

    if not bboxes:
        for obj in objects:
            obj['column'] = 1
        sample['column_number'] = 1
        sample['multi_column_top'] = 0
        sample['multi_column_bottom'] = 1025
        return sample

    # 按x坐标分组
    x_groups = defaultdict(list)
    for bbox in bboxes:
        x_groups[bbox[0]].append(bbox)

    # 检测有效列
    valid_columns = []
    potential_columns = []
    for x, group_bboxes in sorted(x_groups.items()):
        for bbox1 in group_bboxes:
            is_valid = any(is_intersect(bbox1, bbox2, 'y') and not is_intersect(bbox1, bbox2, 'x')
                           for other_x, other_group_bboxes in x_groups.items() if x != other_x
                           for bbox2 in other_group_bboxes)

            if is_valid:
                valid_columns.append(bbox1)
            else:
                potential_columns.append(bbox1)

    # 确定最终的有效列
    valid_columns.sort(key=lambda b: b[1])
    potential_columns.sort(key=lambda b: b[1])

    final_valid_columns = valid_columns[:]

    if valid_columns:
        top_y = min(bbox[1] for bbox in valid_columns)
        bottom_y = max(bbox[1] + bbox[3] for bbox in valid_columns)

        for invalid_bbox in reversed([bbox for bbox in potential_columns if bbox[1] < top_y]):
            if invalid_bbox[2] > 600:
                break
            if any(is_similar_column(invalid_bbox, valid_bbox) for valid_bbox in final_valid_columns):
                final_valid_columns.append(invalid_bbox)
                top_y = invalid_bbox[1]

        for invalid_bbox in [bbox for bbox in potential_columns if bbox[1] + bbox[3] > bottom_y]:
            if invalid_bbox[2] > 600:
                break
            if any(is_similar_column(invalid_bbox, valid_bbox) for valid_bbox in final_valid_columns):
                final_valid_columns.append(invalid_bbox)
                bottom_y = invalid_bbox[1] + invalid_bbox[3]

        middle_bboxes = [bbox for bbox in potential_columns if top_y <= bbox[1] < bottom_y]
        for i, bbox in enumerate(middle_bboxes):
            if bbox[2] > 600:
                last_valid_index = next((j for j in range(i - 1, -1, -1) if middle_bboxes[j][2] <= 600), -1)
                if last_valid_index != -1:
                    bottom_y = middle_bboxes[last_valid_index][1] + middle_bboxes[last_valid_index][3]
                    final_valid_columns = [bbox for bbox in final_valid_columns if bbox[1] + bbox[3] <= bottom_y]
                break


    # 收集有效列的信息
    column_info = defaultdict(lambda: {'top': float('inf'), 'bottom': 0, 'max_width': 0})
    for bbox in final_valid_columns:
        x, y, w, h = bbox
        column_info[x]['top'] = min(column_info[x]['top'], y)
        column_info[x]['bottom'] = max(column_info[x]['bottom'], y + h)
        column_info[x]['max_width'] = max(column_info[x]['max_width'], w)

    # 计算列数和分配列编号
    sorted_columns = sorted(column_info.items())
    column_num = 1
    x_to_column = {}
    column_max_right_x = {}
    column_start_x = {}

    for i, (current_x, info) in enumerate(sorted_columns):
        x_to_column[current_x] = column_num
        current_max_right_x = current_x + info['max_width']

        if column_num not in column_max_right_x or current_max_right_x > column_max_right_x[column_num]:
            column_max_right_x[column_num] = current_max_right_x

        if column_num not in column_start_x:
            column_start_x[column_num] = current_x

        if i < len(sorted_columns) - 1:
            next_x, _ = sorted_columns[i + 1]
            if column_max_right_x[column_num] < next_x:
                column_num += 1

    # 为每个对象分配列编号
    for obj in objects:
        if obj['category_id'] in (0, 3, 6, 7, 8, 9):
            obj_x = round(obj['bbox'][0])
            obj['column'] = next((col for col, start_x in column_start_x.items()
                                  if start_x <= obj_x < column_max_right_x[col]), 1)
        else:
            obj['column'] = 1

    # 计算多列区域的上下边界
    if column_info:
        sample['multi_column_top'] = min(info['top'] for info in column_info.values())
        sample['multi_column_bottom'] = max(info['bottom'] for info in column_info.values())
    else:
        sample['multi_column_top'] = 0
        sample['multi_column_bottom'] = 0

    sample['column_number'] = column_num

    return sample

def sort_pdf_objects(sample):
    objects = sample['objects']
    multi_column_top = sample['multi_column_top']
    multi_column_bottom = sample['multi_column_bottom']

    # 将对象分成三个部分
    top_objects = []
    middle_objects = []
    bottom_objects = []

    for obj in objects:
        y = round(obj['bbox'][1])
        if y < multi_column_top:
            top_objects.append(obj)
        elif y >= multi_column_top and y < multi_column_bottom:
            middle_objects.append(obj)
        else:
            bottom_objects.append(obj)

    # 定义排序键
    def sort_key1(obj):
        return (obj['bbox'][1], obj['bbox'][0])
    def sort_key(obj):
        return (obj['column'], obj['bbox'][1], obj['bbox'][0])

    # 对每个部分进行排序
    top_objects.sort(key=sort_key1)
    middle_objects.sort(key=sort_key)
    bottom_objects.sort(key=sort_key1)

    # 合并排序后的对象
    sorted_objects = top_objects + middle_objects + bottom_objects

    # 添加排序ID
    for i, obj in enumerate(sorted_objects):
        obj['sort_id'] = i

    # 更新sample中的objects
    sample['objects'] = sorted_objects

    return sample


def crop_and_convert_to_bytes(image, bbox):
    # 裁剪图片
    left, top, width, height = bbox
    cropped_image = image.crop((left, top, left + width, top + height))

    # 将裁剪后的图片转换为字节
    img_byte_arr = io.BytesIO()
    cropped_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def convert_to_parquet(dataset, output_dir='./output', batch_size=1000, row_group_size=1000):
    TARGET_FILE_SIZE = 300 * 1024 * 1024  # 300 MB
    block_counter = {}
    file_counter = 0
    current_data = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    category_name = ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.select(range(i, min(i+batch_size, len(dataset))))

        for item in batch:
            doc_key = item['doc_name']
            if doc_key not in block_counter:
                block_counter[doc_key] = 0

            current_time = datetime.now().strftime("%Y%m%d")
            extended_fields = {
                'doc_category': item['doc_category'],
                'collection': item['collection'],
                'page_no': item['page_no'],
                'width': item['width'],
                'height': item['height']
            }

            original_image = item['image']

            for obj in item['objects']:
                block_id = block_counter[doc_key]
                image_content = crop_and_convert_to_bytes(original_image, obj['bbox'])
                block_type = category_name[obj['category_id']]
                extended_fields['bbox'] = obj['bbox']

                row = {
                    '实体ID': item['doc_name'],
                    '块ID': block_id,
                    '时间': current_time,
                    '扩展字段': json.dumps(extended_fields),
                    '文本': obj['text'],
                    '图片': image_content,
                    'OCR文本': json.dumps(item['cells']),
                    '音频': None,
                    'STT文本': None,
                    '其它块': None,
                    '块类型': block_type,
                    'md5': hashlib.md5(item['doc_name'].encode()).hexdigest(),
                    '页ID': item['page_no']
                }

                current_data.append(row)
                block_counter[doc_key] += 1

        # 批量处理后估算大小
        if len(current_data) >= batch_size:
            df = pd.DataFrame(current_data)
            current_size = df.memory_usage(deep=True).sum()

            if current_size >= TARGET_FILE_SIZE:
                output_file = f"{output_dir}/output_part_{file_counter}.parquet"
                df.to_parquet(output_file, engine='pyarrow', row_group_size=row_group_size)
                print(f"Parquet file written: {output_file}, size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")
                current_data = []
                file_counter += 1

    # 处理剩余的数据
    if current_data:
        df = pd.DataFrame(current_data)
        output_file = f"{output_dir}/output_part_{file_counter}.parquet"
        df.to_parquet(output_file, engine='pyarrow', row_group_size=row_group_size)
        print(f"Final Parquet file written: {output_file}, size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")

original_dataset = load_dataset("localDocLayNet.py",local_path="./")
all_subsets = [original_dataset['train'], original_dataset['validation'], original_dataset['test']]
merged_ds = concatenate_datasets(all_subsets)
processed_merged_ds = merged_ds.map(process_sample)
processed_merged_ds_with_column_num = processed_merged_ds.map(determine_column_layout)
processed_merged_ds_sorted = processed_merged_ds_with_column_num.map(sort_pdf_objects)
processed_merged_ds_page_sorted = processed_merged_ds_sorted.sort(["doc_name", "page_no"])
convert_to_parquet(processed_merged_ds_page_sorted)

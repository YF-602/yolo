#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量删除txt文件最后一列数据
适用于格式：数字 小数 小数 小数 小数 小数
"""

import os
import glob
import shutil

def remove_last_column(folder_path):
    """
    删除所有txt文件中每行的最后一列数据
    
    参数:
        folder_path: 包含txt文件的文件夹路径
    """
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not txt_files:
        print("❌ 未找到任何txt文件！")
        return 0
    
    processed_count = 0
    
    for file_path in txt_files:
        try:
            modified_lines = []
            
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 处理每一行
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    modified_lines.append("")
                    continue
                
                # 按空格分割列
                columns = line.split()
                
                if len(columns) <= 1:
                    # 如果只有一列或空行，保持不变
                    modified_lines.append(line)
                    print(f"⚠️  文件 {os.path.basename(file_path)} 第 {line_num} 行只有 {len(columns)} 列，保持不变")
                else:
                    # 删除最后一列
                    new_columns = columns[:-1]
                    new_line = " ".join(new_columns)
                    modified_lines.append(new_line)
            
            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(modified_lines))
            
            print(f"✅ 已处理: {os.path.basename(file_path)} (共 {len(lines)} 行)")
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 处理失败: {os.path.basename(file_path)} - 错误: {e}")
    
    print(f"\n🎉 处理完成！共处理了 {processed_count} 个文件")
    return processed_count

def remove_last_column_with_backup(folder_path, backup_suffix="backup"):
    """
    删除最后一列数据，将处理后的文件保存在backup文件夹中，原文件保持不变
    """
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not txt_files:
        print("❌ 未找到任何txt文件！")
        return 0
    
    # 创建备份文件夹（用于存放处理后的文件）
    backup_dir = os.path.join(folder_path, backup_suffix)
    os.makedirs(backup_dir, exist_ok=True)
    print(f"📁 处理后文件将保存在: {backup_dir}")
    
    processed_count = 0
    
    for file_path in txt_files:
        try:
            filename = os.path.basename(file_path)
            # 处理后的文件路径（在backup文件夹中，文件名不变）
            processed_file_path = os.path.join(backup_dir, filename)
            
            # 读取原文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 处理内容（删除最后一列）
            modified_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    modified_lines.append("")
                    continue
                
                columns = line.split()
                if len(columns) > 1:
                    new_columns = columns[:-1]
                    new_line = " ".join(new_columns)
                    modified_lines.append(new_line)
                else:
                    modified_lines.append(line)
            
            # 将处理后的内容写入backup文件夹中的文件
            with open(processed_file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(modified_lines))
            
            # 统计信息
            original_columns = len(lines[0].strip().split()) if lines else 0
            new_columns = len(modified_lines[0].split()) if modified_lines and modified_lines[0] else 0
            
            print(f"✅ 已处理: {filename}")
            print(f"   📊 列数: {original_columns} → {new_columns}")
            print(f"   💾 处理后文件: {backup_suffix}/{filename}")
            print(f"   📄 原文件保持不变: {filename}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 处理失败: {os.path.basename(file_path)} - 错误: {e}")
    
    print(f"\n🎉 处理完成！共处理了 {processed_count} 个文件")
    print(f"📂 处理后文件保存在: {backup_dir}")
    print(f"📄 原文件保持不变")
    return processed_count

def preview_changes(folder_path):
    """
    预览更改而不实际修改文件
    """
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not txt_files:
        print("❌ 未找到任何txt文件！")
        return
    
    print("🔍 预览模式（不会实际修改文件）:")
    print("=" * 50)
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                continue
                
            print(f"\n📄 文件: {os.path.basename(file_path)}")
            print("修改前 (前3行示例):")
            for i, line in enumerate(lines[:3]):
                line = line.strip()
                if line:
                    columns = line.split()
                    print(f"  第{i+1}行: {line} (共{len(columns)}列)")
            
            print("修改后 (前3行示例):")
            for i, line in enumerate(lines[:3]):
                line = line.strip()
                if line:
                    columns = line.split()
                    new_columns = columns[:-1] if len(columns) > 1 else columns
                    new_line = " ".join(new_columns)
                    print(f"  第{i+1}行: {new_line} (共{len(new_columns)}列)")
                    
        except Exception as e:
            print(f"❌ 读取失败: {os.path.basename(file_path)} - 错误: {e}")

# 使用示例
if __name__ == "__main__":
    folder_path = "./output/labels/test"  # 修改为您的文件夹路径
    
    print("请选择操作模式:")
    print("1. 预览更改（不修改文件）")
    print("2. 直接处理文件")
    print("3. 处理并创建备份")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        preview_changes(folder_path)
        
    elif choice == "2":
        confirm = input("⚠️  确定要直接修改文件吗？(y/N): ").strip().lower()
        if confirm == 'y':
            remove_last_column(folder_path)
        else:
            print("操作已取消")
            
    elif choice == "3":
        remove_last_column_with_backup(folder_path)
        
    else:
        print("无效选择！")
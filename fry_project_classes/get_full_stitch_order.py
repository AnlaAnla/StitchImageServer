

def get_full_stitch_order(row_num: int, column_num: int,debug=False) -> dict:
    """
    生成图像拼接顺序
    Args:
        row_num: 总行数
        column_num: 总列数
    Returns:
        dict: 拼接顺序字典，每个key是拼接步骤序号，value是一个元组(轮数, 图片1, 图片2, 拼接方向, 结果)
    """
    result_dict = {}
    stitch_idx = [1]  # 使用列表以便在递归中修改
    round_num = [1]   # 使用列表以便在递归中修改
    
    # 初始化图像矩阵
    initial_matrix = []
    for i in range(row_num):
        row = []
        for j in range(column_num):
            row.append(str(i * column_num + j + 1))
        initial_matrix.append(row)
    
    if debug:
        print("\nInitial matrix:")
        for row in initial_matrix:
            print(row)

    def recursive_stitch(matrix, direction="horizontal"):
        """
        递归执行拼接操作
        Args:
            matrix: 当前待处理的矩阵
            direction: 当前的拼接方向 ("horizontal" 或 "vertical")
        Returns:
            处理后的矩阵
        """
        # 基本情况：矩阵只有一个元素
        if len(matrix) == 1 and len(matrix[0]) == 1:
            return matrix

        new_matrix = []
        has_stitching = False

        if direction == "horizontal":
            # 水平拼接
            for row in matrix:
                new_row = []
                for i in range(0, len(row), 2):
                    if i + 1 < len(row):
                        # 两两水平拼接
                        new_name = f"{row[i]}_{row[i+1]}"
                        result_dict[stitch_idx[0]] = (round_num[0], row[i], row[i+1], 
                                                    direction, new_name)
                        stitch_idx[0] += 1
                        new_row.append(new_name)
                        has_stitching = True
                    else:
                        new_row.append(row[i])
                new_matrix.append(new_row)
        else:  # vertical
            # 垂直拼接
            max_cols = max(len(row) for row in matrix)
            transposed_matrix = []
            
            # 构建转置矩阵
            for col in range(max_cols):
                col_elements = []
                for row in matrix:
                    if col < len(row):
                        col_elements.append(row[col])
                transposed_matrix.append(col_elements)
            
            # 处理每一列（现在是转置后的行）
            result_matrix = []
            for col in transposed_matrix:
                new_col = []
                for i in range(0, len(col), 2):
                    if i + 1 < len(col):
                        new_name = f"{col[i]}_{col[i+1]}"
                        result_dict[stitch_idx[0]] = (round_num[0], col[i],
                                                    col[i+1], direction, new_name)
                        stitch_idx[0] += 1
                        new_col.append(new_name)
                        has_stitching = True
                    else:
                        new_col.append(col[i])
                result_matrix.append(new_col)
            
            # 转置回来
            new_matrix = [[row[i] for row in result_matrix if i < len(row)] 
                         for i in range(max(len(row) for row in result_matrix))]

        # 如果本轮有拼接操作，增加轮数并打印当前矩阵
        if has_stitching:
            round_num[0] += 1
            if debug:
                print(f"\n当前的方向为：{direction}")
                print(f"\nAfter Round {round_num[0]} :")
                for row in new_matrix:
                    print(row)


        # 判断是否为最后一轮（只剩下一行多列）
        if len(new_matrix) == 1 and len(new_matrix[0]) > 1:
            # 最后一轮，水平拼接剩余的所有元素
            final_row = new_matrix[0]
            final_result = []
            for i in range(0, len(final_row), 2):
                if i + 1 < len(final_row):
                    new_name = f"{final_row[i]}_{final_row[i+1]}"
                    result_dict[stitch_idx[0]] = (round_num[0], final_row[i],
                                                final_row[i+1], "horizontal", new_name)
                    stitch_idx[0] += 1
                    final_result.append(new_name)
                else:
                    final_result.append(final_row[i])
            new_matrix = [final_result]
            if debug:
                print(f"\nFinal Round {round_num[0]} (horizontal):")
                print(new_matrix)
            return new_matrix

        # 确定下一轮的拼接方向
        next_direction = "vertical" if direction == "horizontal" else "horizontal"

        # 继续递归
        if len(new_matrix) > 1 or len(new_matrix[0]) > 1:
            return recursive_stitch(new_matrix, next_direction)
        return new_matrix

    # 开始递归拼接
    recursive_stitch(initial_matrix)
    return result_dict



def test_stitch_images():
    # 测试2×2的情况
    print("Testing 2×2:")
    result = get_full_stitch_order(2, 2)
    print("\nStitch steps:")
    for step, (round_num, img1, img2, direction, result_name) in result.items():
        print(f"Step {step} (Round {round_num}): {img1} + {img2} ({direction}) -> {result_name}")
    
    print("\n" + "="*50 + "\nTesting 3×3:")
    result = get_full_stitch_order(3, 3)
    print("\nStitch steps:")
    for step, (round_num, img1, img2, direction, result_name) in result.items():
        print(f"Step {step} (Round {round_num}): {img1} + {img2} ({direction}) -> {result_name}")
    
    print("\n" + "="*50 + "\nTesting 4×6:")
    result = get_full_stitch_order(4, 6)
    print("\nStitch steps:")
    for step, (round_num, img1, img2, direction, result_name) in result.items():
        print(f"Step {step} (Round {round_num}): {img1} + {img2} ({direction}) -> {result_name}")

    print("\n" + "="*50 + "\nTesting 7×5:")
    result = get_full_stitch_order(7, 5,debug=True)
    print("\nStitch steps:")
    for step, (round_num, img1, img2, direction, result_name) in result.items():
        print(f"Step {step} (Round {round_num}): {img1} + {img2} ({direction}) -> {result_name}")


if __name__ == "__main__":
    test_stitch_images()
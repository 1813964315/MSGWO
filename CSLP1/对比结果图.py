import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = 'C.xlsx'  # 请替换为你的Excel文件路径
sheet_name = 'Sheet6'  # 请替换为你的工作表名称

# 使用pandas读取Excel文件
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 将DataFrame转换为二维数组
C = df.values.tolist()

# 绘制收敛曲线
plt.rcParams['font.family'] = 'Times New Roman'

def plot_lines_with_larger_fonts_and_thicker_lines(curves, labels, colors, final_curve, final_label, final_color):
    # 定义不同的线条样式和标记
    line_styles = [
        ('--', 'o'),  # 虚线 + 圆形标记
        ('-.', 's'),  # 点线 + 正方形标记
        (':', '^'),   # 点划线 + 三角形标记
        ('-', 'v'),   # 实线 + 向下三角形标记
        ('--', '<'),  # 虚线 + 向左三角形标记
        ('-.', '>'),  # 点线 + 向右三角形标记
        (':', 'p'),   # 点划线 + 五边形标记
        ('-', '*'),   # 实线 + 星形标记
        ('--', 'h'),  # 虚线 + 六边形标记
        ('-.', 'D'),  # 点线 + 菱形标记
        (':', 'x')    # 点划线 + 叉形标记
    ]

    # 创建图像，调整图像大小
    plt.figure(figsize=(30, 20))  # 增大图像尺寸

    # 绘制所有曲线（除最终曲线外）
    for curve, label, color, (linestyle, marker) in zip(curves, labels, colors, line_styles):
        plt.plot(curve, label=label, color=color, linewidth=6, linestyle=linestyle, marker=marker, markersize=12)  # 使用不同的线条样式和标记

    # 绘制最终曲线（红色实线）
    plt.plot(final_curve, label=final_label, color=final_color, linewidth=6, linestyle='-', markersize=12)  # 实线 + 圆形标记

    # 设置坐标轴标签和标题
    plt.xlabel('Iteration', fontsize=72)  # 增大字体大小
    plt.ylabel('Total Distance (km)', fontsize=72)
    plt.title('luohuqu', fontsize=72)

    # 设置图例
    plt.legend(fontsize=56, loc='upper right')  # 增大图例字体大小

    # 设置坐标轴刻度字体大小
    plt.xticks(fontsize=64)
    plt.yticks(fontsize=64)

    # 增加外边框的明显性
    ax = plt.gca()  # 获取当前轴
    ax.spines['top'].set_linewidth(3)  # 设置边框宽度
    ax.spines['top'].set_color('black')  # 设置边框颜色
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(3)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_linewidth(3)
    ax.spines['right'].set_color('black')

    # 保存图像，去除白边
    plt.savefig('luouhuau.png', dpi=120, bbox_inches='tight')

    # 显示图像
    plt.show()

# 使用示例数据运行函数
curves = C[:-1]
labels = ['WOA', 'PSO', 'GWO', 'HO', 'BSLO', 'APO', 'RBMO', 'IVY', 'BKA', 'GA', 'SA']
colors = ['purple', 'blue', 'green', 'orange', 'cadetblue', 'm', 'pink', 'brown', 'cyan', 'gold', 'mediumaquamarine']
final_curve = C[-1]
final_label = 'MSGWO'
final_color = 'red'

# 调用函数绘制图像
plot_lines_with_larger_fonts_and_thicker_lines(curves, labels, colors, final_curve, final_label, final_color)
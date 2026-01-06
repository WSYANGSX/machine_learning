import json
from rich.json import JSON
from rich.console import Console
from rich.table import Table
from rich.pretty import Pretty  # <--- 关键引入
from rich import box

console = Console()
table = Table(title="User Data", show_lines=True, box=box.ROUNDED)

table.add_column("Name", style="cyan", no_wrap=True, vertical="middle")
table.add_column("Nested Info")

# 模拟数据
my_dict = {"Math": 90, "English": 85}
my_list = ["Coding", "Swimming", "Reading"]

# 错误写法 (会导致 NotRenderableError):
# table.add_row("Bob", my_dict)

# 正确写法 1: 使用 Pretty (保留颜色和格式，最推荐)
table.add_row("Bob", Pretty(my_dict))

# 正确写法 2: 也可以用来包装列表
table.add_row("Alice", JSON(json.dumps(my_list)))

# 正确写法 3: 如果你只想要普通文本，没有任何高亮
table.add_row("Charlie", str(my_dict))
data = {"id": 1, "status": "active"}
json_str = json.dumps(data)

table.add_row("Server 1", JSON(json_str))
console.print(table)


# from rich import box
# from rich.console import Console
# from rich.table import Table

# console = Console()

# # --- 情况 A: 默认情况 (显示表头，样式为加粗洋红) ---
# table_a = Table(
#     title="Table A: show_header=True",
#     show_header=True,  # <--- 显示列名
#     header_style="bold magenta",  # <--- 列名变成 紫色+加粗
#     box=box.ROUNDED,  # <--- 表格边框样式为 圆角
# )
# table_a.add_column("ID")
# table_a.add_column("Name", justify="center")
# table_a.add_row("1", "Alice")

# # --- 情况 B: 隐藏表头 ---
# table_b = Table(
#     title="Table B: show_header=False",
#     show_header=False,  # <--- 彻底隐藏列名行
#     # header_style 在这里失效，因为表头都不显示了
# )
# table_b.add_column("ID")
# table_b.add_column("Name")
# table_b.add_row("1", "Alice")

# console.print(table_a)
# print("\n" + "-" * 30 + "\n")  # 分隔线
# console.print(table_b)
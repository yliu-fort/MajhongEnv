def print_mahjong_tiles():
    """打印所有麻将牌的 Unicode 字符。"""
    for codepoint in range(0x1F000, 0x1F030):  # 遍历麻将牌的 Unicode 区段
        try:
            char = chr(codepoint)
            print(f"U+{codepoint:04X}: {char}", end="  ") # 格式化输出
            if (codepoint - 0x1F000 + 1) % 8 == 0: # 每行打印8个
                print()
        except ValueError:
            print(f"U+{codepoint:04X}: Invalid Unicode code point")

print_mahjong_tiles()

haiDisp = ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", 
           "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", 
           "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", 
           "东", "南", "西", "北", "白", "发", "中")

haiDispEmojis = tuple(range(0x1F007, 0x1F010)) \
                + tuple(range(0x1F019, 0x1F022)) \
                + tuple(range(0x1F010, 0x1F019)) \
                + tuple(range(0x1F000, 0x1F007))

def color_and_bold_emoji(emoji, color_code):
    # ANSI escape code for bold and colored text
    return f"\033[0;{color_code}m{emoji}\033[0m"

mahjong_tiles = [chr(codepoint) for codepoint in range(0x1F000, 0x1F030)]
colored_emojis = [color_and_bold_emoji(tile, 31 + i % 6) for i, tile in enumerate(mahjong_tiles)]

# Print colored and bold emojis
for i, emoji in enumerate(colored_emojis):
    if i % 8 == 0 and i != 0:  # Break line every 8 emojis for better visualization
        print()
    print(emoji, end=" ")
print()

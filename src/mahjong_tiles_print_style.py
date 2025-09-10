class MahjongTileStyle:
    _style_text = ("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", 
           "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", 
           "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", 
           "东", "南", "西", "北", "白", "发", "中")

    _style_emoji = tuple(chr(x) for x in tuple(range(0x1F007, 0x1F010)) \
                    + tuple(range(0x1F019, 0x1F022)) \
                    + tuple(range(0x1F010, 0x1F019)) \
                    + tuple(range(0x1F000, 0x1F004)) \
                    + (0x1F006, 0x1F005, 0x1F004))
    
    @staticmethod
    def get_style(hai):
        return f"{MahjongTileStyle._style_emoji[hai]} " if hai != 33 else \
            MahjongTileStyle._style_emoji[hai]
    
    @staticmethod
    def get_colored_style(hai, color_code=31):
    # ANSI escape code for bold and colored text
        return f"\033[0;{color_code}m{MahjongTileStyle._style_emoji[hai]}\033[0m "

    @staticmethod
    def get_tile_printout(tile_id: int) -> str:
        if tile_id == -1:
            return "🀫 "
        assert 0 <= tile_id < 136
        hai = tile_id // 4
        str = MahjongTileStyle.get_colored_style(hai) \
            if tile_id == 16 or tile_id == 52 or tile_id == 88 else \
            MahjongTileStyle.get_style(hai)
        return str
    
    @staticmethod
    def get_tiles_printout(tile_ids: list) -> list:
        return "".join([MahjongTileStyle.get_tile_printout(tile_id) for tile_id in tile_ids])


def tile_printout(tile_id: int):
    return MahjongTileStyle.get_tile_printout(tile_id)

def tiles_printout(tile_ids: list, sort=True):
    return MahjongTileStyle.get_tiles_printout(sorted(tile_ids)) if sort else \
            MahjongTileStyle.get_tiles_printout(tile_ids)


if __name__ == "__main__":
    [print((emoji), end=" ") for emoji in MahjongTileStyle._style_emoji]
    print('--------------------------------')
    print(MahjongTileStyle.get_tiles_printout(range(136)))
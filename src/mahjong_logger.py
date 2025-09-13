from typing import List, Dict
import xml.etree.ElementTree as ET

from tenhou_to_mahjong import TenhouRoundTracker


def encode_meld(meld):
    base = meld["m"][0] // 4
    offset = meld["offset"]
    match meld["type"]:
        case "chi":
            called = meld["m"].index(meld["claimed_tile"])
            base_and_called = ((base // 9) * 7 + base % 9) * 3 + called
            t0 = meld["m"][0] - base * 4
            t1 = meld["m"][1] - (base + 1) * 4
            t2 = meld["m"][2] - (base + 2) * 4
            return (base_and_called << 10) | (t2 << 7) | (t1 << 5) | (t0 << 3) | (1 << 2) | offset
        case "pon"|"chakan":
            called = sorted(meld["m"][:3]).index(meld["claimed_tile"])
            base_and_called = base * 3 + called
            is_kan = meld["type"] == "chakan"
            t4 = (meld["m"][-1] % 4) if is_kan else [x for x in range(4) if x not in [y % 4 for y in meld["m"]]][0]
            return (base_and_called << 9) | (t4 << 5) | is_kan << 4 | (not is_kan) << 3 | offset
        case "kan":
            if meld["opened"]:
                called = meld["claimed_tile"] % 4
                base_and_called = base * 4 + called % 4
                return (base_and_called << 8) | offset
            else:
                base_and_called = base * 4
                return base_and_called << 8
            
class MahjongLoggerBase:

    draw_syms = "TUVW"
    discard_syms = "DEFG"

    def __init__(self, seed):
        self.root = ET.Element("mjloggm")
        self.root.set("ver", "2.3")
        self.add_shuffle_tag(seed)
        self.add_go_tag()
        self.add_un_tag()
        self.add_taikyoku_tag()
    
    def add_shuffle_tag(self, seed):
        shuffle_tag = ET.SubElement(self.root, "SHUFFLE")
        shuffle_tag.set("seed", "mt19937ar-sha512-n288-base64,"+seed)
        shuffle_tag.set("ref", "")
    
    def add_go_tag(self, go_type=169, lobby=0):
        # GO type=169: 凤南赤
        go_tag = ET.SubElement(self.root, "GO")
        go_tag.set("type", str(go_type))
        go_tag.set("lobby", str(lobby))

    def add_un_tag(self, players=["Asan", "Bsan", "Csan", "Dsan"]):
        un_tag = ET.SubElement(self.root, "UN")
        for i, player in enumerate(players):
            un_tag.set(f"n{i}", player)
        un_tag.set("dan", "0,0,0,0")
        un_tag.set("rate", "0.0,0.0,0.0,0.0")
        un_tag.set("sx", "M,M,F,F")
    
    def add_taikyoku_tag(self, oya=0):
        taikyoku_tag = ET.SubElement(self.root, "TAIKYOKU")
        taikyoku_tag.set("oya", str(oya))

    def add_init(self, round, kyoutaku, dice, dora_indicator, scores, oya, hands):
        init_tag = ET.SubElement(self.root, "INIT")
        init_tag.set("seed", f"{round[0]},{round[1]},{kyoutaku},{dice[0]},{dice[1]},{dora_indicator[0]}")
        init_tag.set("ten", ",".join([str(x) for x in scores]))
        init_tag.set("oya", str(oya))
        for i, hand in enumerate(hands):
            init_tag.set(f"hai{i}", ",".join([str(x) for x in hand]))

    def add_draw(self, who:int, tile:int):
        ET.SubElement(self.root, self.draw_syms[who]+str(tile))

    def add_discard(self, who:int, tile:int):
        ET.SubElement(self.root, self.discard_syms[who]+str(tile))
    
    # 记录天凤格式log e.g., <N who="2" m="25611" />
    def add_meld(self, who:int, meld):
        meld_tag = ET.SubElement(self.root, "N")
        meld_tag.set("who", str(who))
        meld_tag.set("m", str(encode_meld(meld)))
        
    # 记录天凤格式log e.g., <DORA hai="79" />
    def add_dora(self, tile:int):
        dora_tag = ET.SubElement(self.root, "DORA")
        dora_tag.set("hai", str(tile))
    
    # 输出天凤格式的log. <REACH who="3" step="1"/> <REACH who="3" ten="250,250,250,240" step="2"/>
    def add_reach_declare(self, who:int):
        reach_tag = ET.SubElement(self.root, "REACH")
        reach_tag.set("who", str(who))
        reach_tag.set("step", "1")
    
    def add_reach_accepted(self, who:int, scores):
        reach_tag = ET.SubElement(self.root, "REACH")
        reach_tag.set("who", str(who))
        reach_tag.set("ten", ",".join([str(x) for x in scores]))
        reach_tag.set("step", "2")
    
    '''
        # 记录天凤格式log <AGARI ba="1,1" hai="1,3,5,6,8,11,16,17" m="20935,51242" machi="3" ten="30,11700,0" yaku="20,1,34,2,54,1" doraHai="91" who="2" fromWho="2" sc="230,-40,210,-40,220,130,330,-40" />
    meld_log = f''
    if self.agari['m']:
        meld_log = f'm="{",".join([str(encode_meld(m)) for m in self.agari["m"]])}" '
    ura_log = f''
    if self.ura_indicator:
        ura_log = f'doraHaiUra="{",".join([str(x) for x in self.ura_indicator])}" '
    yakuman_log = f''
    if self.agari['yakuman_tenhou']:
        yakuman_log = f'yakuman="{",".join([str(x) for x in self.agari["yakuman_tenhou"]])}" '
    log += f'<AGARI ba="{self.round[1]},{self.num_riichi}" hai="{",".join([str(x) for x in self.hands[self.agari["who"]]])}" ' + \
            meld_log + \
            f'machi="{self.agari["machi"]}" ten="{",".join([str(x) for x in self.agari["ten"]])}" yaku="{",".join([str(x) for x in self.agari["yaku_tenhou"]])}" ' + \
            yakuman_log + \
            f'doraHai="{",".join([str(x) for x in self.dora_indicator])}" ' + \
            ura_log + \
            f'who="{self.agari["who"]}" fromWho="{self.agari["fromwho"]}" sc="{",".join([str(x) for sc in zip(self.scores, self.score_deltas) for x in sc])}"'
    '''
    def add_agari(self, round, num_riichi, hands, melds, machi, ten, scores, score_deltas, yaku, dora_indicator, who, fromwho, ura_indicator=None, yakuman=None, pao=None):
        agari_tag = ET.SubElement(self.root, "AGARI")
        agari_tag.set("ba", f"{round[1]},{num_riichi}")
        agari_tag.set("hai", ",".join([str(x) for x in hands]))
        if melds:
            agari_tag.set("m", ",".join([str(encode_meld(m)) for m in melds]))
        agari_tag.set("machi", str(machi))
        agari_tag.set("ten", ",".join([str(x) for x in ten]))
        agari_tag.set("yaku", ",".join([str(x) for x in yaku]))
        if yakuman:
            agari_tag.set("yakuman", ",".join([str(x) for x in yakuman]))
        agari_tag.set("doraHai", ",".join([str(x) for x in dora_indicator]))
        if ura_indicator:
            agari_tag.set("doraHaiUra", ",".join([str(x) for x in ura_indicator]))
        agari_tag.set("who", str(who))
        agari_tag.set("fromWho", str(fromwho))
        agari_tag.set("sc", ",".join([str(x) for sc in zip(scores, score_deltas) for x in sc]))
        if pao:
            agari_tag.set("paoWho", str(pao))

        return agari_tag

    # 记录天凤格式log
    def add_ryuukyoku(self, round, num_riichi, scores, score_deltas, hands, tenpai, num_players, type=None):
        ryuukyoku_tag = ET.SubElement(self.root, "RYUUKYOKU")
        if type:
            ryuukyoku_tag.set("type", type)
        ryuukyoku_tag.set("ba", f"{round[1]},{num_riichi}")
        ryuukyoku_tag.set("sc", ",".join([str(x) for sc in zip(scores, score_deltas) for x in sc]))
        for p in range(num_players):
            if tenpai[p]:
                ryuukyoku_tag.set(f'hai{p}', ",".join([str(x) for x in sorted(hands[p])]))

        return ryuukyoku_tag
                                      
    def add_owari(self, prev_tag: ET.SubElement, scores, score_deltas):
        prev_tag.set("owari", ",".join([str(x) for sc in zip(scores, score_deltas) for x in sc]))

    def write_to_file(self, file_path):
        tree = ET.ElementTree(self.root)
        with open(file_path, "wb") as file:
            tree.write(file, encoding="utf-8", xml_declaration=True)

    def __str__(self):
        return ET.tostring(self.root, encoding="unicode", method="xml")


class MahjongLogger(MahjongLoggerBase, TenhouRoundTracker):
    def add_init(self, round, kyoutaku, dice, dora_indicator, scores, oya, hands: List):
        super().add_init(round, kyoutaku, dice, dora_indicator, scores, oya, hands)
        seed_list = [round[0],round[1],kyoutaku,dice[0],dice[1],dora_indicator[0]]
        TenhouRoundTracker.start_init(self, seed_list, oya, {i: pai for i, pai in enumerate(hands)})

    def add_draw(self, who:int, tile:int):
        super().add_draw(who, tile)
        TenhouRoundTracker.draw(self, who, tile)

    def add_discard(self, who:int, tile:int):
        super().add_discard(who, tile)
        TenhouRoundTracker.discard(self, who, tile)
    
    def add_meld(self, who:int, meld):
        super().add_meld(who, meld)
        TenhouRoundTracker.apply_meld(self, who, encode_meld(meld))

    def add_dora(self, tile:int):
        super().add_dora(tile)
        TenhouRoundTracker.add_dora(self, tile)
    
    def add_reach_declare(self, who:int):
        super().add_reach_declare(who)
        TenhouRoundTracker.reach(self, who, 1)

    def add_reach_accepted(self, who:int, scores):
        super().add_reach_accepted(who, scores)
        TenhouRoundTracker.reach(self, who, 2)
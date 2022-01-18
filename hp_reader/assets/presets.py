"""
presets

All size tuples are in order of (width, height)
"""

class Preset():
    def __init__(self, 
        width:int,
        height:int,
        icon_pos:list[tuple[int,int]],
        icon_name:str,
        icon_size:tuple[int,int],
        hpbar_pos:list[tuple[int,int]],
    ):
        self.width = width
        self.height = height
        self.icon_pos = icon_pos
        self.icon_pos_hw = [
            (icon_pos[0][1],icon_pos[0][0]),
            (icon_pos[1][1],icon_pos[1][0])
        ]
        self.icon_name = icon_name
        self.icon_size = icon_size
        self.icon_size_hw = (icon_size[1],icon_size[0])
        self.hpbar_pos = hpbar_pos
        self.hpbar_pos_hw = [
            (hpbar_pos[0][1],hpbar_pos[0][0]),
            (hpbar_pos[1][1],hpbar_pos[1][0])
        ]
        self.hp_h_st = hpbar_pos[0][1]
        self.hp_h_ed = hpbar_pos[1][1]
        hp_w = hpbar_pos[1][0] - hpbar_pos[0][0]
        self.hp_w_st = hpbar_pos[0][0] + int(hp_w*0.27)
        self.hp_w_ed = hpbar_pos[0][0] + int(hp_w*0.70)


UHD110COM = Preset(**{
    'width':3840,
    'height':2160,
    'icon_pos':[(1065, 120),(1170,210)],
    'icon_name':'assets/uhd_110.png',
    'icon_size':(105,90),
    'hpbar_pos':[(1220,135),(2660,190)]
})

FHD100BOS = Preset(
    width=1920,
    height=1080,
    icon_pos=[(635,60),(685,90)],
    icon_name='assets/fhd_100.png',
    icon_size=(50,30),
    hpbar_pos=[(710,62),(1235,85)]
)
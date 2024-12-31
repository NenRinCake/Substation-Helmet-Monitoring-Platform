import pypinyin

def chinese_name_to_english(chinese_name):
    # 将中文姓名分割成姓和名
    last_name = chinese_name[0]
    first_name = chinese_name[1:]

    # 将姓转换为拼音并取第一个字母大写
    english_last_name = pypinyin.lazy_pinyin(last_name)[0].capitalize()

    # 将名转换为拼音并取每个拼音的首字母大写
    english_first_name = ''.join([pinyin.capitalize() for pinyin in pypinyin.lazy_pinyin(first_name)]).title()

    # 输出英文姓名
    english_name = english_last_name + ' ' + english_first_name
    return english_name

# 测试
chinese_name = input("请输入中文姓名：")


hint_list = [
    "급여",
    "일부본인부담",
    "전액본인부담",
    "본인부담금",
    "공단부담금",
    "선택진료료",
    "선택진료료외",
    "비급여",
    "요양급여",
]

keyword2hint = {
    "전액": "전액본인부담",
    "비급여": "비급여",
    "요양": "요양급여",
    "일부본인부담": "일부본인부담",
    "전액본인부담": "전액본인부담",
}

def main():
    if len(hint_list) != len(set(hint_list)):
        from collections import Counter
        c = Counter(hint_list)
        print(c.most_common(3))

if __name__ == '__main__':
    main()

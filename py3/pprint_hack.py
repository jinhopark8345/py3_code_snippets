
def demo1():
    import pprint
    pp = pprint.PrettyPrinter(width=80, compact=True)
    pp.pprint(examples[0].__dict__)

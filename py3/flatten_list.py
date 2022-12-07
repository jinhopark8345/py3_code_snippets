
def flatten_list(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten_list(S[0]) + flatten_list(S[1:])
    return S[:1] + flatten_list(S[1:])
